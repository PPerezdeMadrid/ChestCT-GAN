#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Comprobar python3.11
if ! command_exists python3.11; then
    echo "Error: python3.11 no está instalado o no está en PATH. Es necesario para ejecutar todo el proceso."
    exit 1
fi

if ! command_exists npm; then
    echo "Error: npm no está instalado. Necesario para lanzar la app Node.js."
    exit 1
fi

echo "Usando únicamente python3.11 y un entorno .venv"
echo "Continuando..."

# Clonar repositorio si no existe
if [ ! -d "../ChestCT-GAN" ]; then
    cd ..
    echo "Clonando repositorio ChestCT-GAN..."
    git clone https://github.com/PPerezdeMadrid/ChestCT-GAN.git
fi

cd ChestCT-GAN

# Crear entorno .venv con python3.11 si no existe
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi

# Activar entorno
source .venv/bin/activate

# Instalar pip y dependencias base
pip3.11 install --upgrade pip

# Verificar si ya existen los datos
DATA_PATH="src/Pipeline/Data/manifest-160866918333/Lung-PET-CT-Dx"
if [ -d "$DATA_PATH" ]; then
    echo "Los datos ya están descargados en $DATA_PATH. Omitiendo descarga..."
else
    echo "Datos no encontrados. Procediendo con la descarga..."

    # Instalar nbiatoolkit y ejecutar descarga
    pip3.11 install nbiatoolkit

    cd src/Pipeline/Data
    python NBIA_download.py
    cd ../../..
fi

# Instalar dependencias del pipeline
pip3.11 install -r src/Pipeline/requirements.txt

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
cd src/interfazChestGAN
npm install
nohup npm start > node_app.log 2>&1 &
cd ../..

# Ejecutar pipeline
cd src/Pipeline
python main_pipeline.py run --ip_frontend http://127.0.0.1:8080
