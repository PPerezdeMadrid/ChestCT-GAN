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
