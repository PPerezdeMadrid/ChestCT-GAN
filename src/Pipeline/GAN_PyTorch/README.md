# Generación de Imágenes con GANs

Este proyecto permite entrenar, generar y evaluar modelos GAN (Generative Adversarial Networks) para la generación de imágenes. Se incluyen implementaciones de **DCGAN** y **WGAN**.

---

## Instalación

Antes de comenzar, asegúrese de tener instaladas todas las dependencias necesarias. Puede hacerlo con:

```bash
pip install -r requirements.txt
```

---

## Entrenar el modelo

Para entrenar un modelo GAN, ejecute uno de los siguientes comandos:

```bash
python train.py --model dcgan
python train.py --model wgan
```

Si no se especifica el modelo, por defecto se entrenará **DCGAN**.

---

## Generar Imágenes

Para generar imágenes utilizando un modelo preentrenado:

```bash
python generate.py --model dcgan --load_path ./model_checkpoint.pth --num_output 128
```

Parámetros por defecto:
- **model:** `dcgan`
- **load_path:** `generated_images/model_ChestCT.pth`
- **num_output:** `64`

---

## Evaluar Modelo

Para evaluar el modelo, utilice el siguiente comando:

```bash
python eval_model.py --load_path ..\..\model\model_wgan\model_epoch_990.pth
```

Si no se especifica `load_path`, el valor por defecto es:
- `generated_images/model_ChestCT.pth`

---

## Generar Gráfico de Pérdidas

Para visualizar la evolución de las pérdidas del generador y el discriminador durante el entrenamiento:

```bash
python3 graphLogs.py --csv training_log_wgan.csv
```

# Estructura de archivos 

Aquí tienes el README en español con la estructura de archivos de tu proyecto:  

---

# **README: Estructura del Proyecto**  

## **Descripción del Proyecto**  
Este proyecto está diseñado para el entrenamiento y evaluación de Redes Generativas Adversarias (GANs) aplicadas a imágenes médicas, específicamente radiografías de tórax. La estructura del repositorio incluye scripts para entrenamiento, evaluación, optimización de hiperparámetros y ejecución de la tubería de procesamiento.  

---

## **Estructura de Directorios**  

## **Directorio Principal: `Pipeline/`**  
- **`Data/`** - Carpeta para almacenar datasets.  
- **`GAN_PyTorch/`** - Contiene scripts y archivos relacionados con GANs.  
- **`Metaflow/`** - Probablemente almacena scripts de automatización del flujo de trabajo.  
- **`README.md`** - Documentación del proyecto.  
- **`pipeline.md`** - Documentación sobre la tubería de procesamiento.  
- **`main_pipeline.py`** - Script principal de ejecución del pipeline.  
- **`requirements.txt`** - Lista de dependencias necesarias.  
- **`training_log_dcgan_*.csv`** - Registros de entrenamiento de distintos experimentos con DCGAN.  
- **`Pipeline.png`** - Imagen representativa de la tubería de procesamiento.  

---

## **Directorio `GAN_PyTorch/`**  
Contiene scripts y archivos específicamente relacionados con el entrenamiento y evaluación de modelos GAN.  

- **Modelos GAN**  
  - `dcgan.py` - Implementación de Deep Convolutional GAN (DCGAN).  
  - `wgan.py` - Implementación de Wasserstein GAN (WGAN).  

- **Scripts de Entrenamiento**  
  - `train.py` - Script principal de entrenamiento de GANs.  
  - `train_pipeline.py` - Script para entrenar modelos dentro del pipeline de MLOPS.  

- **Evaluación y Utilidades**  
  - `eval_model.py` - Script para evaluar modelos entrenados.  
  - `generate.py` - Generación de imágenes con modelos entrenados.  
  - `graphLogs.py` - Generación de gráficos a partir de los logs de entrenamiento.  
  - `optimizeHyperparam.py` - Optimización de hiperparámetros.  
  - `utils.py` - Funciones auxiliares.  

- **Registros de Entrenamiento**  
  - `training_log_dcgan_*.csv` - Logs de entrenamiento de DCGAN.  
  - `training_log_wgan_*.csv` - Logs de entrenamiento de WGAN.  

- **Configuración y Setup**  
  - `config.json` - Archivo de configuración.   
  - `requirements.txt` - Dependencias necesarias.  
  - `README.md` - Documentación específica de GANs.  

