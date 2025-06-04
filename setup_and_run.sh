#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Comprobamos python3.11 y python3.13
if ! command_exists python3.11; then
    echo "Error: python3.11 no está instalado o no está en PATH. Es necesario para descargar los datos."
    exit 1
fi

if ! command_exists python3.13; then
    echo "Error: python3.13 no está instalado o no está en PATH. Es necesario para ejecutar el pipeline."
    exit 1
fi

echo "Se requiere python3.11 para la descarga y python3.13 para el pipeline."
echo "Continuando..."

# Clonar repositorio si no existe
if [ ! -d "ChestCT-GAN" ]; then
    git clone https://github.com/PPerezdeMadrid/ChestCT-GAN.git
fi

cd ChestCT-GAN

# Crear entorno python3.13 para el pipeline si no existe
if [ ! -d ".venv_py313" ]; then
    python3.13 -m venv .venv_py313
fi

# Crear entorno python3.11 para descarga si no existe
if [ ! -d ".venv_py311" ]; then
    python3.11 -m venv .venv_py311
fi

# Activar entorno python3.11 para descargar e instalar nbiatoolkit
source .venv_py311/bin/activate
pip install --upgrade pip
pip install nbiatoolkit

# Ejecutar script de descarga con python3.11
cd src/Pipeline/Data
python NBIA_download.py
cd ../../..

# Desactivar entorno python3.11
deactivate

# Activar entorno python3.13 para pipeline
source ..venv_py313/bin/activate

# Instalar dependencias para pipeline
pip install --upgrade pip
pip install -r src/Pipeline/requirements.txt

# Crear archivo .env en interfazChestGAN si no existe
ENV_PATH="src/interfazChestGAN/.env"
if [ ! -f "$ENV_PATH" ]; then
    echo "Creando archivo .env en src/interfazChestGAN"
    cat <<EOL > "$ENV_PATH"
AWS_ACCESS_KEY_ID=to_do
AWS_SECRET_ACCESS_KEY=to_do
AWS_REGION=to_do
SESSION_SECRET=to_do
ADMIN_PASSWD=admin
EOL
else
    echo "El archivo .env ya existe en src/interfazChestGAN"
fi

# Lanzar app Node.js en background
cd interfazChestGAN
npm install
nohup npm start > node_app.log 2>&1 &
cd ..

# Ejecutar pipeline con python3.13
cd src/Pipeline
python main_pipeline.py run --ip_frontend http://127.0.0.1:8080