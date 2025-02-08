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