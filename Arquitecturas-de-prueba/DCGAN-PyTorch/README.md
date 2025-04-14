# DCGAN

Este proyecto implementa una **DCGAN (Deep Convolutional Generative Adversarial Network)** en PyTorch para generar imágenes sintéticas a partir de conjuntos de datos médicos, en particular tomografías computarizadas (CT) del tórax.

## 📌 ¿Qué es una DCGAN?

Una **DCGAN** es una variante de las redes generativas adversarias (GANs) que utiliza capas convolucionales profundas. Las GANs están compuestas por dos redes que compiten entre sí:

- **Generador (Generator):** genera imágenes falsas que imitan las reales.
- **Discriminador (Discriminator):** intenta distinguir entre imágenes reales y falsas.

Durante el entrenamiento, ambos modelos mejoran simultáneamente: el generador aprende a engañar al discriminador, y el discriminador se vuelve mejor diferenciando. En el caso de una DCGAN, se utilizan arquitecturas convolucionales profundas para capturar mejor las características visuales de las imágenes.

Este enfoque se ha vuelto especialmente útil en medicina, donde la **falta de grandes volúmenes de datos etiquetados** limita el entrenamiento de modelos robustos. Al generar imágenes sintéticas realistas, se puede enriquecer el dataset y mejorar los algoritmos de diagnóstico.

---

## 🗂 Estructura del Proyecto

```
DCGAN-PyTorch/
├── ChestTC_dcgan_*.gif     # GIF del entrenamiento cada ciertos epochs
├── config.json             # Configuración base para entrenamiento
├── dcgan.py                # Arquitectura principal del modelo DCGAN
├── dcgan512.py             # Variante para imágenes de 512x512
├── train.py                # Script para entrenar el modelo
├── generate.py             # Script para generar imágenes nuevas
├── eval_model.py           # Evaluación del modelo entrenado
├── graphLogs.py            # Visualización de métricas de entrenamiento
├── requirements_dcgan.txt        # Dependencias necesarias
├── README.md               
└── ...
```

---

## 🚀 Uso del Proyecto

### Entrenamiento de un modelo

```bash
python train.py --model dcgan --dataset chestct
```

Parámetros disponibles:
- `--model`: puede ser `dcgan` o `wgan` (si implementado).
- `--dataset`: opciones disponibles: `chestct` o `nbia`.

El script tomará la configuración del archivo `config.json`, dependiendo del modelo que elijas modificar.

### Generación de imágenes nuevas

```bash
python generate.py -load_path checkpoints/modelo_final.pth -num_output 10
```

Parámetros:
- `-load_path`: ruta al modelo entrenado (checkpoint).
- `-num_output`: número de imágenes a generar.
- `-compare`: si se activa, compara imágenes reales con las generadas.



## 📊 Evaluación y Visualización

### Evaluación del Modelo

Puedes evaluar un modelo utilizando el script `eval_model.py`. Este script permite evaluar la calidad de un modelo GAN utilizando diversas métricas, como la precisión del discriminador y generador, SSIM, PSNR y LPIPS. 

### Uso:

```bash
python eval_model.py --dataset <dataset> --model_name <model_name>
```

### Argumentos:
- `--dataset`: El conjunto de datos a utilizar para la evaluación. Puede ser uno de los siguientes:
  - `nbia` (por defecto)
  - `chestct`
- `--model_name`: El nombre del modelo o checkpoint que quieres evaluar. Ejemplo: `model_ChestCT.pth`.

Los resultados de la evaluación incluyen la precisión del discriminador, la confianza del generador y las métricas SSIM, PSNR y LPIPS.



### Visualización de Logs de Entrenamiento

Puedes visualizar los logs de entrenamiento con el script `graphLogs.py`. Este script genera un gráfico a partir de los logs de entrenamiento, permitiéndote observar el rendimiento del modelo a lo largo del tiempo.

#### Uso:

```bash
python graphLogs.py --log_file <log_file.csv>
```

### Argumentos:
- `--log_file`: El archivo CSV con los logs de entrenamiento. Ejemplo: `training_log_dcgan_2025-03-23.csv`.

El gráfico generado incluye la pérdida del discriminador, la pérdida del generador y otras métricas de rendimiento.


### Logs de Entrenamiento

Los resultados de entrenamiento se pueden encontrar en el archivo `training_log_dcgan_12Feb.csv`, el cual contiene detalles sobre el progreso del entrenamiento, como las métricas de pérdida y precisión a lo largo de las épocas.

Este archivo puede ser usado como entrada para `graphLogs.py` para obtener visualizaciones de los resultados del entrenamiento.

