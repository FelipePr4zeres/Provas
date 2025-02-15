import os
from datetime import datetime
from provas import parse_pdf_questions
from config import JSON_DIR, IMAGES_DIR

def process_pdf(pdf_path):
    """
    Processa um PDF e salva os resultados com timestamp
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    # Cria nome do arquivo baseado no timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{base_name}_{timestamp}.json"
    output_path = os.path.join(JSON_DIR, output_filename)

    # Processa o PDF
    output_data = parse_pdf_questions(pdf_path, output_folder=IMAGES_DIR)

    # Salva o resultado
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processamento concluído. Resultado salvo em: {output_path}")
    print(f"Imagens extraídas salvas em: {IMAGES_DIR}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python main.py caminho/para/arquivo.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
