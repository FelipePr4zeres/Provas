#!/bin/bash

# Atualiza os repositórios
sudo apt-get update

# Instala ferramentas de compilação necessárias
sudo apt-get install -y build-essential
sudo apt-get install -y python3-dev
sudo apt-get install -y rust-all
sudo apt-get install -y cargo

# Instala o Tesseract e suas dependências
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y tesseract-ocr-por
sudo apt-get install -y tesseract-ocr-eng

# Atualiza pip
python3 -m pip install --upgrade pip

# Instala wheel primeiro
pip install --upgrade wheel setuptools

# Instala as dependências Python
pip install -r requirements.txt

# Verifica a instalação
tesseract --version
