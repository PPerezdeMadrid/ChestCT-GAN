# ChestCT-GAN
Proyecto de fin de carrera: Generación de tomografías de pulmones con cáncer para generar datos sintéticos y mejorar los algoritmos de detección de cáncer

# Falta por escribir mucho


## Progreso
**DC-GAN**
- Con hiperparámetros comunes: 
![Hiperparámetros 1](img/dcgan_hiperparam1.png)
- Con ajuste de hiperparámetros
    + Reducción de la tasa de aprendizaje a 0.001
    + Aumento de ngf de 64 a 128 (número de filtros del generador, aumenta la capacidad de capturar características más complejas)
    + Aumento de ndf de 64 a 128 (número de filtros del Discriminador)
![Hiperparámetros 1](img/dcgan_hiperparam2.png)

Ejemplo de imágenes en el epoch **90**:
![epoch90_dcgan](img/epoch90_dcgan.png)


**WGAN**
- Con ajuste de hiperparámetros
[!WGAN](wganGraph.png)
Ejemplo de imágenes en el epoch **90**:
![epoch90_dcgan](img/epoch90_wgan.png)
