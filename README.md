Readme pendiente de escribir
# Generación de Imágenes Sintéticas de Tomografías Computarizadas de Pulmones con Cáncer con una GAN

Este repositorio contiene el código y la documentación del Trabajo de Fin de Grado (TFG) de Paloma Pérez de Madrid. El proyecto se centra en la generación de imágenes sintéticas de tomografías computarizadas de pulmones con cáncer mediante una Red Generativa Antagónica (GAN). 

## Estructura del Repositorio

```
.
├── Arquitecturas-de-prueba   # Tres arquitecturas evaluadas para seleccionar la DCGAN
├── README.md                 
├── X-Ray-pneumonia           # Anexo sobre la generación de radiografías con neumonía
├── doc                       # Documentación adicional
├── img                       # Imágenes de referencia y ejemplos generados
└── src                       # Código fuente del proyecto
    ├── pipeline              # Implementación del pipeline de MLOps con Metaflow
    └── interfazChestGAN      # Interfaz de usuario en Node.js
```

## Descripción del Proyecto
Este proyecto busca generar imágenes médicas sintéticas para ayudar en el entrenamiento de modelos de detección de cáncer de pulmón. Se emplea una arquitectura GAN optimizada para la creación de tomografías computarizadas sintéticas, con un enfoque en la aplicabilidad en entornos médicos y de investigación.

## Tecnologías Utilizadas
- **PyTorch** para la implementación de la GAN
- **Metaflow** para la gestión del pipeline de MLOps
- **Node.js** para el desarrollo de la interfaz de usuario

## Instalación y Uso
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   ```
2. Instala las dependencias necesarias:
   ```bash
   cd src/pipeline
   pip install -r requirements.txt
   ```
   ```bash
   cd src/interfazChestGAN
   npm install
   ```
3. Ejecución del sistema:
   - Para ejecutar la interfaz de usuario en un servidor local:
     ```bash
     cd src/interfazChestGAN
     npm start
     ```
   - Para ejecutar el pipeline de entrenamiento en un entorno local:
     ```bash
     cd src/pipeline
     python main_pipeline.py
     ```
   - En la demostración final, la interfaz se ejecutará en una instancia **EC2 de AWS**, y el pipeline en un **clúster de Kubernetes**. Ambas funciones pueden ejecutarse de manera independiente.

## Fuentes de Datos
Los datos de entrenamiento utilizados en este proyecto provienen de las siguientes fuentes:

- **Lung-PET-CT-Dx**:  
  Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020).  
  *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set].*  
  The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461  

- **Chest CT Scan Images (Kaggle)**:  
  https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images  

Se han seguido las **políticas de uso de datos y citación** requeridas por cada fuente.

## Contacto
Para más información, puedes contactarme a través de mi perfil de GitHub.

