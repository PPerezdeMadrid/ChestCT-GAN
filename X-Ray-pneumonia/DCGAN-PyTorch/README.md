

Modelo de clasificación: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download


Las primeras épocas se ven bien: Loss_D es razonablemente baja (~0.3-0.4) y Loss_G es alta (~7-9), lo que indica que el generador está aprendiendo.

Inestabilidad alrededor de la época 27: Un Loss_D de 3.62 y un Loss_G de 13.7 sugieren que el discriminador perdió el equilibrio y dejó que el generador lo engañara fácilmente.

Degradación a partir de la época 34-50: Se observa que Loss_D aumenta significativamente, alcanzando valores por encima de 1, lo que indica que el discriminador está teniendo problemas para diferenciar entre muestras reales y falsas.

Sobreajuste y oscilaciones: Desde la época 50 en adelante, parece que el entrenamiento entra en una fase de alta variabilidad. Esto puede ser un síntoma de un colapso de modo en el generador o de que el discriminador se está volviendo demasiado fuerte.


nz: lo aumentamos a 200 Un tamaño de vector latente más grande puede permitir que el generador tenga más espacio para aprender y generar una variedad más amplia de imágenes.

