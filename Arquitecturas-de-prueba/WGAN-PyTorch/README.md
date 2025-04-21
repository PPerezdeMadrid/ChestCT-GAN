# Implementación de WGAN-GP

Este apartado contiene la implementación de una Red Generativa Antagónica de Wasserstein con Penalización por Gradiente (WGAN-GP) para la generación de imágenes. El modelo puede entrenarse utilizando penalización por gradiente o recorte de pesos. El código incluye varios scripts para entrenar, generar y evaluar imágenes, así como para gestionar las configuraciones y los registros de entrenamiento.

## Requisitos

Asegúrate de tener instalados los siguientes paquetes de Python:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-image`

Puedes instalarlos utilizando `pip`:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-image
```

## Estructura del Proyecto

```plaintext
├── config.json                # Configuración de hiperparámetros y rutas
├── training_log_wgan.csv      # Registros de entrenamiento
├── generate.py                # Generación de imágenes
├── eval_model.py              # Evaluación del modelo
├── graphLogs.py               # Graficar registros de entrenamiento
├── model.py                   # Definición del modelo WGAN-GP
└── utils.py                   # Funciones utilitarias
```

## Descripción de los Scripts

### `generate.py`

Este script se usa para generar imágenes a partir de un modelo entrenado. Permite cargar un modelo preentrenado, especificar la cantidad de imágenes a generar y, opcionalmente, comparar las imágenes generadas con las reales.

**Uso:**

```bash
python generate.py -load_path <ruta_checkpoint> -num_output 64 -compare
```

- `-load_path`: Ruta al checkpoint del modelo entrenado (por defecto: `./model/model.pth`)
- `-num_output`: Número de imágenes generadas (por defecto: 64)
- `-compare`: Opción para mostrar una comparación entre las imágenes generadas y las reales.

### `eval_model.py`

Este script evalúa el rendimiento del modelo utilizando varias métricas como la precisión del discriminador, la precisión del generador, SSIM, PSNR y LPIPS.

**Uso:**

```bash
python eval_model.py --dataset chestct --model_name model_ChestCT.pth
```

- `--dataset`: Elige entre los datasets `"nbia"` o `"chestct"` para la evaluación.
- `--model_name`: Nombre del checkpoint del modelo a cargar.

Las métricas de evaluación incluyen:
- **Precisión del discriminador**
- **Precisión del generador**
- **SSIM** (Índice de Similaridad Estructural)
- **PSNR** (Relación de Señal a Ruido Pico)
- **LPIPS** (Similitud Perceptual de Parche de Imagen Aprendida)

### `graphLogs.py`

Este script permite graficar los registros de entrenamiento guardados en `training_log_wgan.csv`, visualizando métricas como la pérdida del generador y la pérdida del discriminador a lo largo de las épocas.

### `config.json`

Este archivo contiene la configuración del modelo, incluidos los hiperparámetros, las rutas para guardar los modelos e imágenes generadas, y las rutas de los datasets de entrada.

```json
{
    "params": {
        "bsize": 128,
        "imsize": 64,
        "nc": 1,
        "nz": 100,
        "ngf": 64,
        "ndf": 64,
        "nepochs": 1000,
        "lr": 0.0001,
        "beta1": 0.5,
        "save_epoch": 100,
        "critic_iters": 5
    },
    "model": {
        "path": "model/model_wgan",
        "image_path": "images/images_wgan"
    },
    "datasets":{
        "chestKaggle": "../../../../TFG/ChestCTKaggle/Data/",
        "nbia": "../../src/Pipeline/Data/Data-Transformed"
    }
}
```

- **params**: Define los hiperparámetros del modelo (por ejemplo, `latent_dim`, `batch_size`, `num_epochs`, `learning_rate`).
- **model**: Establece las rutas para guardar el modelo y las imágenes generadas.
- **datasets**: Establece las rutas de los conjuntos de datos.

### `training_log_wgan.csv`

Este archivo CSV guarda los registros de cada época durante el entrenamiento. Incluye métricas como la pérdida del discriminador, la pérdida del generador y la distancia Wasserstein. Puedes usar `graphLogs.py` para visualizar estas métricas.

### Opciones de Entrenamiento del Modelo

Puedes entrenar el modelo utilizando Penalización por Gradiente o Recorte de Pesos.

- **`train_gp.py`**: Entrena el modelo WGAN-GP con penalización por gradiente.
- **`train_wp.py`**: Entrena el modelo WGAN con recorte de pesos (menos efectivo según el artículo original de WGAN).

### Justificación de la Elección de `train_gp` sobre `train_wp`

El uso de **penalización por gradiente** en WGAN-GP es más efectivo que el **recorte de pesos** en la mejora de la estabilidad durante el entrenamiento, como se argumenta en el artículo original de WGAN-GP (Arjovsky et al., 2017). En el artículo, se señala que el recorte de pesos puede inducir una distorsión en la función de optimización, lo que provoca que la convergencia sea más lenta y, en algunos casos, inestable. Por otro lado, la **penalización por gradiente** mejora la estabilidad del entrenamiento al suavizar el gradiente y evitar los problemas derivados de la saturación del discriminador.

El uso de **WGAN-GP** ha demostrado ser más eficiente en la generación de imágenes de alta calidad y en la mejora de las métricas de evaluación, como se observa en las comparaciones de los resultados experimentales. Por lo tanto, es recomendable usar `train_gp.py` para entrenar el modelo, ya que ofrece una mejor convergencia y estabilidad en comparación con `train_wp.py`.

### Comando de Ejemplo para Entrenar con Penalización por Gradiente

```bash
python train_gp.py
```

Esto entrenará el modelo utilizando los hiperparámetros especificados en `config.json` y guardará los checkpoints durante el proceso.