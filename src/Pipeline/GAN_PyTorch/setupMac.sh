#!/bin/bash

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Verificar la versión de torch
python3 -c "import torch; print(torch.__version__)"

pip install -r requirements.txt
