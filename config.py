import os

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "extracted_images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# Configurações do modelo
MODEL_NAME = "microsoft/layoutlmv3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configurações do OCR
TESSERACT_CMD = "/usr/bin/tesseract"  # Caminho padrão no Linux

# Criar diretórios necessários
for directory in [OUTPUT_DIR, IMAGES_DIR, JSON_DIR]:
    os.makedirs(directory, exist_ok=True)
