# Usa una imagen base de Python
FROM python:3.12

# Configura el directorio de trabajo
WORKDIR /app

# Copia el contenido del directorio actual al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el archivo principal
CMD ["python", "python Data/NBIA_download.py"]
CMD ["python", "main_pipeline.py", "run"]
