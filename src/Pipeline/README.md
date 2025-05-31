# 🧬 Pipeline de MLOps con Metaflow y Kubernetes / MLOps Pipeline with Metaflow and Kubernetes

Este repositorio contiene un pipeline completo de generación de imágenes médicas sintéticas mediante GANs, gestionado con **Metaflow**, desplegable en **Kubernetes**, y preparado para producción con subida a **AWS S3**.

This repository contains a complete pipeline for generating synthetic medical images using GANs, managed with **Metaflow**, deployable on **Kubernetes**, and production-ready with upload to **AWS S3**.



## Índice / Table of Contents

### 🇪🇸 Español
- [1. Instrucciones para ejecutar el pipeline](#1-instrucciones-para-ejecutar-el-pipeline)
  - [1.1 Antes de empezar](#1-antes-de-empezar)
  - [1.2 Ejecutar el pipeline](#2-ejecutar-el-pipeline)
- [2. Opciones del pipeline](#⚙️-opciones-del-pipeline)
- [3. Estructura del directorio](#🗂️-estructura-del-directorio)
- [4. Diagrama del pipeline (Metaflow)](#🔁-diagrama-del-pipeline-metaflow)

### 🇬🇧 English
- [1. Instructions to run the pipeline](#1-instructions-to-run-the-pipeline)
  - [1.1 Before you start](#11-before-you-start)
  - [1.2 Run the pipeline](#12-run-the-pipeline)
- [2. Pipeline options](#2-pipeline-options)
- [3. Directory structure](#3-directory-structure)
- [4. Pipeline diagram (Metaflow)](#4-pipeline-diagram-metaflow)



## 🇪🇸 1. Instrucciones para ejecutar el pipeline

### 1.1 Antes de empezar

🔹 **Descarga del Dataset NBIA (TCIA)**  
Descarga los datos desde:  
📎 [https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)  
Archivo: `Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia`

> 💡 Si tienes problemas con NBIA Data Retriever, puedes usar el script `Data/NBIA_download.py` o la librería `nbiatoolkit` en Python.

🔹 **Configura la ruta del dataset en `GAN_PyTorch/config.json`**  
Modifica la siguiente sección para que apunte a los datos descargados:

```json
"datasets": {
    "chestKaggle": "../../../../TFG/ChestCTKaggle/Data/",
    "nbia": "Data/manifest-160866918333"
}
````

🔹 **Configura tus credenciales de AWS**
El paso `upload_files_cloud` requiere tener AWS configurado. Ejecuta:

```bash
aws configure
```



### 1.2 Ejecutar el pipeline

```bash
cd src/Pipeline
python main_pipeline.py run
```

🔎 Para ver logs en tiempo real:

```bash
tail -f pipeline_live.log
```



## Opciones del pipeline

```bash
python main_pipeline.py run [opciones]
```

| Flag            | Descripción                                       | Valor por defecto          |
| --------------- | ------------------------------------------------- | -------------------------- |
| `--dataset`     | Dataset a utilizar (`nbia` o `chestct`)           | `nbia`                     |
| `--ip_frontend` | IP del frontend que recibirá las imágenes         | `www.chestgan.tech`        |
| `--model_type`  | Tipo de modelo GAN (`dcgan` o `wgan`)             | `dcgan`                    |
| `--num_output`  | Número de imágenes sintéticas a generar           | `100`                      |
| `--yaml_path`   | Ruta al archivo de pesos para evaluación          | `GAN_PyTorch/weights.yaml` |
| `--tag`         | Añadir etiquetas a la ejecución (puede repetirse) | —                          |
| `--max-workers` | Número máximo de procesos paralelos               | `16`                       |

> 💡 Ver más opciones:

```bash
python main_pipeline.py run --help
```



## Estructura del directorio

```bash
📂 Pipeline
│-- 📂 Data                      
│-- 📂 GAN_PyTorch               
│-- 📂 model                     # *
│-- 📂 images                    # *
│-- 📂 evaluation                # *
│-- 📄 pipeline_live.log         
│-- 📄 README.md                 
│-- 📄 requirements.txt          
│-- 📄 pipeline.md               
│-- 📄 template_EvalModel.md     
│-- 📄 main_pipeline.py          
│-- 📂 kubernetes                
```

>  \* Los directorios marcados con \* se generan durante la ejecución.



## Diagrama del pipeline (Metaflow)

```mermaid
graph TD
    start[Start<br/>Selección de Imágenes para el modelo]
    train_model[train_model<br/>Entrenar el modelo]
    eval_model[eval_model<br/>Evaluar el modelo]
    generate_imgs[generate_imgs<br/>Generar Imágenes Sintéticas]
    generate_report[generate_report<br/>Generar un informe mensual]
    upload_files_cloud[upload_files_cloud<br/>Subida de archivos a la nube]
    end[End<br/>Fin del pipeline]

    start --> train_model
    train_model --> eval_model
    eval_model --> generate_imgs
    generate_imgs --> generate_report
    generate_report --> upload_files_cloud
    upload_files_cloud --> end
```

---
---

## 🇬🇧 1. Instructions to run the pipeline

### 1.1 Before you start

🔹 **Download the NBIA Dataset (TCIA)**
Get the data from:
📎 [https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
File: `Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia`

> 💡 If NBIA Data Retriever fails, use `Data/NBIA_download.py` or the `nbiatoolkit` library.

🔹 **Set the dataset path in `GAN_PyTorch/config.json`**
Update this section to match your download location:

```json
"datasets": {
    "chestKaggle": "../../../../TFG/ChestCTKaggle/Data/",
    "nbia": "Data/manifest-160866918333"
}
```

🔹 **Configure your AWS credentials**
The `upload_files_cloud` step needs AWS credentials. Run:

```bash
aws configure
```



### 1.2 Run the pipeline

```bash
cd src/Pipeline
python main_pipeline.py run
```

🔎 To see real-time logs:

```bash
tail -f pipeline_live.log
```



## 2. Pipeline options

```bash
python main_pipeline.py run [options]
```

| Flag            | Description                                        | Default value              |
| --------------- | -------------------------------------------------- | -------------------------- |
| `--dataset`     | Dataset to use (`nbia` or `chestct`)               | `nbia`                     |
| `--ip_frontend` | Frontend IP that will receive the generated images | `www.chestgan.tech`        |
| `--model_type`  | GAN model type (`dcgan` or `wgan`)                 | `dcgan`                    |
| `--num_output`  | Number of synthetic images to generate             | `100`                      |
| `--yaml_path`   | Path to weights file for evaluation                | `GAN_PyTorch/weights.yaml` |
| `--tag`         | Add tags to execution (repeatable)                 | —                          |
| `--max-workers` | Maximum number of parallel processes               | `16`                       |

> 💡 See all options:

```bash
python main_pipeline.py run --help
```

---

## 3. Directory structure

```bash
📂 Pipeline
│-- 📂 Data                      
│-- 📂 GAN_PyTorch               
│-- 📂 model                     # *
│-- 📂 images                    # *
│-- 📂 evaluation                # *
│-- 📄 pipeline_live.log         
│-- 📄 README.md                 
│-- 📄 requirements.txt          
│-- 📄 pipeline.md               
│-- 📄 template_EvalModel.md     
│-- 📄 main_pipeline.py          
│-- 📂 kubernetes                
```

> \* Folders marked with \* are created during pipeline execution.



## 4. Pipeline diagram (Metaflow)

```mermaid
graph TD
    start[Start<br/>Image selection]
    train_model[train_model<br/>Train the model]
    eval_model[eval_model<br/>Evaluate the model]
    generate_imgs[generate_imgs<br/>Generate synthetic images]
    generate_report[generate_report<br/>Generate monthly report]
    upload_files_cloud[upload_files_cloud<br/>Upload files to the cloud]
    end[End<br/>Pipeline end]

    start --> train_model
    train_model --> eval_model
    eval_model --> generate_imgs
    generate_imgs --> generate_report
    generate_report --> upload_files_cloud
    upload_files_cloud --> end
```
