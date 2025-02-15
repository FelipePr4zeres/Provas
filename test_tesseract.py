import pytesseract
from PIL import Image

print("Versão do Tesseract:", pytesseract.get_tesseract_version())
print("Idiomas disponíveis:", pytesseract.get_languages())
