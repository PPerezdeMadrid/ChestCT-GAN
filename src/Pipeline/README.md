# Pipeline de MLOPS con Metaflow y Kubernetes

## Instrucciones para ejecutar el pipeline

### Antes de empezar

1. Asegúrate de tener los archivos descargados del dataset NBIA (TCIA) en tu máquina.
\(https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) El archivo se llama `Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia`. Si no puedes descargarlo con la herramienta *NBIA Data Retriever* prueba a hacerlo con la librería `nbiatoolkit` de python.

2. En el archivo `main_pipeline.py`, localiza la siguiente línea de código:

```python
dataset_path = "../../../../ChestCT-NBIA/manifest-1608669183333"  # CAMBIAR !!
```

3. **¡Importante!** Cambia el valor de `dataset_path` para que apunte a la carpeta donde se encuentran los archivos descargados del dataset.

### Ejecución del pipeline
```bash
cd src/Pipeline
python main_pipeline.py run
```

---

## Estructura del directorio 

```bash
📂 Pipeline
│-- 📂 Data                  # Datos utilizados para entrenamiento
│-- 📂 GAN_PyTorch           # Implementación de las arquitecturas GAN en PyTorch
│-- 📂 model                 # * Modelos entrenados y checkpoints 
│-- 📂 images                # * Imágenes generadas por el modelo
│-- 📂 evaluation            # * Archivos sobre la evaluación de los checkpoints
│-- 📄 README.md             # Este archivo
│-- 📄 requirements.txt      # Dependencias necesarias para ejecutar el proyecto
│-- 📄 pipeline.md           # Descripción de la pipeline de datos
│-- 📄 template_EvalModel.md # Plantilla para la evaluación del modelo
│-- 📄 main_pipeline.py      # Script principal del pipeline de Metaflow
│-- 📄 Pipeline.png          # Diagrama del pipeline

Nota*: Se genera automáticamente con el pipeline
```

```bash
python3 main_pipeline.py show 
Metaflow 2.13.9 executing ChestGAN for user:palomaperezdemadrid



Step start
    Selección de Imágenes para el modelo 
    => train_model

Step train_model
    Entrenar el modelo 
    => eval_model

Step eval_model
    Evaluar el modelo 
    => generate_imgs

Step generate_imgs
    Generar Imágenes Sintéticas 
    => generate_report

Step generate_report
    Generar un informe mensual 
    => end

Step end
    Fin del pipeline.
```

Requisitos: Kubernetes, awscli, ...