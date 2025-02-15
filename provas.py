import os
import fitz
import json
import pytesseract
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pytesseract import Output
from config import MODEL_NAME, DEVICE, TESSERACT_CMD

# Configura caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Inicializa modelo
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def extract_images_from_page(pdf_page, page_index, output_folder="extracted_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_info = []
    image_list = pdf_page.get_images(full=True)
    for idx, img in enumerate(image_list):
        xref = img[0]
        pix = fitz.Pixmap(pdf_page.parent, xref)
        image_path = os.path.join(output_folder, f"page_{page_index}_img_{idx}.png")
        pix.save(image_path)

        # Tentativa de obter as coordenadas do retângulo que contém a imagem
        # Se "block" não for confiável, retornamos apenas image_path
        bbox = None
        blocks_raw = pdf_page.get_text("rawdict")["blocks"]
        for block in blocks_raw:
            if "image" in block and block["image"] is not None:
                if block["image"]["xref"] == xref:
                    bbox = block["bbox"]
                    break

        # bbox em (x1, y1, x2, y2). Caso necessário, converter para
        # algo relativo ou para "acima/abaixo" de determinada linha.
        images_info.append({
            "image_path": image_path,
            "bbox": bbox
        })
        pix = None
    return images_info

def extract_pdf_text_styled(pdf_page):
    """
    Extrai texto e informações de estilo (ex: fonte, flags para bold/italic)
    Usando PyMuPDF, que lê PDFs nativos (não escaneados).
    Retorna lista de spans:
       [{
          "text": "Exemplo",
          "bbox": [x1, y1, x2, y2],
          "font_name": "Helvetica-Bold",
          "font_size": 12.0,
          "style": "bold"/"italic"/"normal"
        }, ... ]
    Obs.: Se for um PDF escaneado (sem camadas de texto), este método pode retornar vazio.
    """
    results = []
    page_dict = pdf_page.get_text("dict")  # blocos + spans
    for block in page_dict["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                bbox = span["bbox"]  # [x1,y1,x2,y2]
                font_name = span.get("font", "")
                font_size = span.get("size", 0)
                flags = span.get("flags", 0)

                # flags (PyMuPDF docs): 2=italic, 20=bold, 4=underline etc.
                # Nem sempre é exato. Alternativamente, verificar "Bold"/"Italic" no font_name.
                style = "normal"
                if "bold" in font_name.lower() or (flags & 2**4):
                    style = "bold"
                if "italic" in font_name.lower() or (flags & 2):
                    style = "italic"

                results.append({
                    "text": text,
                    "bbox": bbox,
                    "font_name": font_name,
                    "font_size": font_size,
                    "style": style
                })
    return results

def extract_ocr_with_tesseract(page_image_path):
    """
    Extrai tokens via Tesseract, com bounding boxes absolutos
    no sistema de coordenadas da imagem.
    Retorna lista de dicts:
      [
        {
          "word": "Exemplo",
          "left": L,
          "top": T,
          "width": W,
          "height": H
        },
        ...
      ]
    """
    data = pytesseract.image_to_data(Image.open(page_image_path), output_type=Output.DICT)
    words_info = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word:  # ignora tokens vazios
            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]
            words_info.append({
                "word": word,
                "left": left,
                "top": top,
                "width": width,
                "height": height
            })
    return words_info

def run_layoutlmv3_on_tokens(page_image_path, words_info):
    """
    Constrói input para LayoutLMv3 a partir dos tokens e bounding boxes do Tesseract.
    Atribui as boxes normalizadas (0-1000) ou outro critério, conforme convenção do LayoutLM.
    Retorna tokens + possíveis labels (se estivermos usando token classification).
    """

    # Exemplo simplificado de normalização de bounding boxes.
    # O LayoutLMv3 espera coords de 0 a 1000. Ajuste conforme a doc.
    im = Image.open(page_image_path)
    w, h = im.size

    tokens = []
    boxes = []
    for wi in words_info:
        x1 = wi["left"]
        y1 = wi["top"]
        x2 = x1 + wi["width"]
        y2 = y1 + wi["height"]

        # Normaliza para 0-1000
        x1_n = int((x1 / w) * 1000)
        y1_n = int((y1 / h) * 1000)
        x2_n = int((x2 / w) * 1000)
        y2_n = int((y2 / h) * 1000)

        tokens.append(wi["word"])
        boxes.append([x1_n, y1_n, x2_n, y2_n])

    encoding = processor(
        im,
        text=tokens,
        boxes=boxes,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**{k: v.to(DEVICE) for k, v in encoding.items()})

    # Se for um modelo fine-tuned para NER ou token-class, poderíamos decodificar labels:
    # predicted_labels = torch.argmax(outputs.logits, dim=2).cpu().numpy().tolist()

    result = []
    for i, token in enumerate(tokens):
        result.append({
            "token": token,
            "bbox": boxes[i],
            # "label": label_map.get(predicted_labels[0][i], "O")  # Exemplo
        })
    return result

def identify_multiple_choice(tokens):
    """
    Identifica questões de múltipla escolha a partir de padrões, ex.:
    - "(A)", "A)", "a)", "A.", letra em parênteses, em bolinha etc.
    Implementar heurísticas personalizadas. Aqui, um exemplo simplificado.
    """
    # Agrega tokens num texto único
    text_full = " ".join([t["token"] for t in tokens])
    # Exemplo de extração simples (apenas 1 questão):
    # Em prática, você separaria enunciado e opções por regex ou regras.
    question_data = {
        "question_text": text_full,
        "options": [],
        "styled_tokens": tokens
    }

    # Exemplo ingênuo de busca de opções:
    # Procurar padronizadamente "A)", "B)", "C)", etc.
    # Dependendo do PDF, pode ser "(a)", "(b)", "1)", etc.
    # A customização depende muito do layout real.
    # Aqui é apenas demonstração.
    return [question_data]

def map_images_to_question(images_info, tokens_data):
    """
    Tenta determinar se a imagem está 'acima' ou 'abaixo' de algum
    trecho de texto, comparando coordenadas. Retorna array com metadados.
    """
    # Supondo bounding boxes do texto normalizado 0-1000, e bbox das imagens
    # em coordenadas originais do PDF. É preciso alinhar sistemas de coords.
    # Se o PDF for a imagem inteira, pode ser mais fácil usar a mesma base.
    # Aqui, exemplificamos de forma simples, assumindo que
    # se a imagem tiver y2 < y1 do enunciado principal, está "acima", etc.

    # Exemplo: definimos enunciado como as primeiras linhas e tentamos
    # ver se a imagem está acima/abaixo delas. Em produção, precisa de heurísticas melhores.
    mapped = []
    for img in images_info:
        if img["bbox"] is None:
            # Se não temos bbox exata, devolvemos sem mapeamento
            mapped.append({
                "image_path": img["image_path"],
                "position_in_question": None
            })
            continue

        x1, y1, x2, y2 = img["bbox"]
        # Heurística tola: se y2 < 100, diz que está 'acima do enunciado'
        if y2 < 100:
            pos = "above"
        else:
            pos = "below"
        mapped.append({
            "image_path": img["image_path"],
            "position_in_question": pos
        })
    return mapped

def parse_pdf_questions(pdf_path, output_folder="extracted_images"):
    """
    Percorre cada página, extrai imagens, textos, rodando OCR + LayoutLMv3
    e tentando inferir questões de múltipla escolha e estilo (bold/italic).
    Retorna estrutura JSON.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_doc = fitz.open(pdf_path)
    result = {"pages": []}

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]

        # Extrai e salva imagens
        images_info = extract_images_from_page(page, page_index, output_folder)

        # Converte página em imagem p/ OCR
        pix = page.get_pixmap(dpi=150)
        page_image_path = os.path.join(output_folder, f"page_{page_index}.png")
        pix.save(page_image_path)

        # OCR para obter tokens + bounding boxes
        words_info = extract_ocr_with_tesseract(page_image_path)

        # Rodar LayoutLMv3 (opcional, dependendo do que precisamos classificar)
        layoutlm_tokens = run_layoutlmv3_on_tokens(page_image_path, words_info)

        # Identifica questões de múltipla escolha (forma simplificada)
        questions_found = identify_multiple_choice(layoutlm_tokens)

        # Mapeia cada questão às imagens (exemplo: 'above' ou 'below')
        for q in questions_found:
            q["images"] = map_images_to_question(images_info, layoutlm_tokens)

        # Extrai estilo (bold/italic) e fonte do texto "nativo" do PDF
        # Caso o PDF não seja nativo (só imagem), será vazio
        styled_spans = extract_pdf_text_styled(page)

        # Monta o output final da página
        page_output = {
            "page_number": page_index + 1,
            "questions": questions_found,
            "styled_spans": styled_spans
        }
        result["pages"].append(page_output)

    pdf_doc.close()
    return result

if __name__ == "__main__":
    pdf_file = "exemplo.pdf"
    output_data = parse_pdf_questions(pdf_file)

    with open("output_questions.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)