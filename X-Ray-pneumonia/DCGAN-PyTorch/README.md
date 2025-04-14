# Anexo para la generación de imágenes de radiografías con neumonía

Este proyecto utiliza un modelo DCGAN (Deep Convolutional Generative Adversarial Network) para generar imágenes sintéticas de radiografías con neumonía. A continuación se detallan los pasos para ejecutar el entrenamiento y generar imágenes.

## Estructura del Proyecto

```bash
DCGAN-PyTorch/
├── README.md                
├── config.json              # Archivo de configuración (rutas, parámetros)
├── dcgan512.py              # Implementación del modelo DCGAN (resolución 512x512)
├── dcgan.py                 # Modelo base de DCGAN (64x64)
├── eval_model.py            # Script para evaluar el modelo
├── generate.py              # Script para generar imágenes
├── graphLogs.py             # Funciones para graficar los logs
├── images_prueba/           # Carpeta para imágenes de prueba generadas
├── model_prueba/            # Carpeta para modelos entrenados
├── requirements_xray.txt   # Requisitos para el proyecto (instalación)
├── train.py                 # Script para entrenar el modelo
├── main.py                  # Script principal para ejecutar el entrenamiento
├── X-Ray_TC_dcgan.gif       # Animación generada del entrenamiento
└── utils.py                 # Funciones de utilidad
```

## Requisitos previos

1. Tener Python 3.6+ instalado.
2. Instalar las dependencias del proyecto. Ejecuta el siguiente comando para instalar todas las librerías necesarias:

```bash
pip install -r requirements_xray.txt
```

## Pasos para ejecutar el entrenamiento

1. **Descargar los datos de Kaggle**  
   Si no tienes los datos descargados, descomenta las siguientes líneas en `main.py` para descargarlos:

   ```python
   # Descargar los datos
   # data_path = download_xray_data()
   # prepare_data(data_path, "../Data_train")
   ```

   Esto descargará automáticamente el conjunto de datos de radiografías de Kaggle y lo preparará para el entrenamiento.

2. **Ejecutar el entrenamiento**  
   Una vez que los datos estén listos, puedes ejecutar el entrenamiento del modelo utilizando el siguiente comando:

   ```bash
   python main.py
   ```

   El modelo comenzará a entrenar, y los resultados se guardarán en las carpetas definidas en `config.json`.

## Archivos y directorios importantes

- **Modelos guardados**: Los modelos entrenados se guardarán en la carpeta `model_prueba/`.
- **Imágenes generadas**: Las imágenes generadas durante el entrenamiento se guardarán en la carpeta `images_prueba/`.
- **Evaluación**: Los resultados de la evaluación se guardarán en la carpeta `evaluation/`.
- **Logs y gráficos**: Los logs del entrenamiento y gráficos se guardarán y se pueden visualizar usando las funciones en `graphLogs.py`.


## Configuración

La configuración del proyecto se define en el archivo `config.json`. En este archivo podrás ajustar las rutas de los datos, las configuraciones del modelo y otros parámetros importantes. Asegúrate de revisar y modificar este archivo según tus necesidades.

## Generar imágenes de 512x512

Si deseas generar imágenes de resolución 512x512 en lugar de la resolución predeterminada, sigue estos pasos:

1. **Modificar la resolución en `config.json`**  
   Abre el archivo `config.json` y cambia el valor de `imsize` a `512`. Además, se recomienda ajustar el tamaño del batch (bsize) a `32` para obtener un mejor rendimiento con esta resolución. El cambio debería verse así:

   ```json
   {
    "params": {
        "bsize": 32,
        "imsize": 512, 
     }
    }
   ```

2. **Modificar el script de entrenamiento**  
   En el archivo `train.py`, cambia la siguiente línea de importación:

   De:

   ```python
   from dcgan import Generator, Discriminator, weights_init
   ```

   A:

   ```python
   from dcgan512 import Generator, Discriminator, weights_init
   ```

   Esto asegurará que se utilice la versión del modelo diseñada para generar imágenes de 512x512.

---

## Scripts

### `generate.py`

Este script se utiliza para generar imágenes a partir de un modelo previamente entrenado.

#### Parámetros:
- **`-load_path`**:  
  Ruta del checkpoint (modelo entrenado) que se va a cargar. El valor predeterminado está configurado como `f'{model_path}/{model_pth}'`, que busca el modelo en la ubicación definida por `model_path` y `model_pth`.  
  - **Tipo**: string  
  - **Descripción**: Especifica la ruta del modelo que se cargará para la generación de imágenes.

- **`-num_output`**:  
  Número de imágenes generadas. Por defecto, este parámetro está establecido en 5, lo que significa que se generarán 5 imágenes.  
  - **Tipo**: entero  
  - **Descripción**: Especifica cuántas imágenes generar al ejecutar el script.

#### Uso:
El script usa el parámetro `-load_path` para cargar un modelo previamente entrenado desde la ruta especificada y genera la cantidad de imágenes indicada con el parámetro `-num_output`.

Ejemplo de ejecución:

```bash
python generate.py -load_path 'ruta/a/modelo.pth' -num_output 5
```

---

### `eval_model.py`

Este script se utiliza para evaluar el rendimiento de un modelo entrenado.

#### Parámetros:
- **`model_name`**:  
  Nombre del archivo del modelo que se va a evaluar. Por defecto, este parámetro está configurado como `"model_epoch_400.pth"`, lo que significa que se evaluará el modelo guardado con ese nombre si no se proporciona otro valor.  
  - **Tipo**: string  
  - **Descripción**: Especifica el archivo del modelo que se cargará para realizar la evaluación.

- **`model_path`**:  
  Ruta donde se encuentra el modelo que se va a evaluar. Este valor se construye concatenando la ruta definida en el archivo de configuración (`config["model"]["path_dcgan"]`) con el nombre del modelo (`model_name`). Por defecto, `model_name` es `"model_epoch_400.pth"`.  
  - **Tipo**: string  
  - **Descripción**: Especifica la ubicación completa del modelo a evaluar.

#### Uso:
El script carga el modelo desde la ruta `model_path` y evalúa su rendimiento en el conjunto de datos de prueba.

Ejemplo de ejecución:

```bash
python eval_model.py
```

Este comando usará la ruta predeterminada para cargar el modelo y evaluarlo. Si deseas evaluar un modelo diferente, puedes modificar el valor de `model_name` directamente en el código o añadir un parámetro en la ejecución (si lo configuras).

---
Nota: 
- Modelo de clasificación: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
- Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download



