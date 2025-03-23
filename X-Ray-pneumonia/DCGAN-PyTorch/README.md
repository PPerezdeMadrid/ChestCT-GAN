

Modelo de clasificación: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download



Anexo para la generación de imágenes de radiografías con pneumonia

para ejecutar el entrenamiento primero debes:
- Si no te has descargado los datos de kaggle descomenta las lineas # Descargar los datos
# data_path = download_xray_data()
# prepare_data(data_path,"../Data_train")) de main.py para descargarte los datos. 

despues haces python main.py y listo, empezará a entrenarse. Las carpetas donde se guardarán los modelos, las imágenes generadas y la evalaución están definidas en config.json

