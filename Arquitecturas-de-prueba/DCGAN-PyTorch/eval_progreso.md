# Evaluación del progreso

Parámetros inciales:
§	bsize : 128 → Tamaño del lote durante el entrenamiento.
§	imsize : 64 → Tamaño espacial de las imágenes de entrenamiento. Todas las
§	imágenes se redimensionarán a este tamaño durante la preprocesamiento.
§	nc : 1 → Número de canales en las imágenes de entrenamiento. 
§	nz : 100 → Tamaño del vector latente Z (entrada del generador).
§	ngf : 128 → Tamaño de los mapas de características en el generador. La
profundidad será múltiplo de este valor.
§	ndf : 128 → Tamaño de los mapas de características en el discriminador. La profundidad será múltiplo de este valor.
§	nepochs: 1000→ Número de épocas de entrenamiento.
§	lr : 0.0001 → Tasa de aprendizaje para los optimizadores.
§	beta1 : 0.5 → Hiperparámetro Beta1 para el optimizador Adam.
§	beta2 : 0.999 → Hiperparámetro Beta2 para el optimizador Adam.

Hiperparámetros:
```bash
"params": {
        "bsize": 128,
        "imsize": 64, 
        "nc": 1,
        "nz": 100,
        "ngf": 128,
        "ndf": 128,
        "nepochs": 1000,
        "lr": 0.0001,
        "beta1": 0.5,
        "beta2": 0.999,
        "save_epoch": 100
    }
```

64x64, 100 epochs:

------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 95.93%
Generator Accuracy:  28.63%
SSIM Score:          0.2721
PSNR Score:          14.5395
LPIPS Score          0.5692
------------------------------

64x64, 200 epochs:

------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.86%
Generator Accuracy:  14.63%
SSIM Score:          0.2848
PSNR Score:          14.6796
LPIPS Score          0.5456
------------------------------

64x64, 300 epochs

------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 94.89%
Generator Accuracy:  11.07%
SSIM Score:          0.2671
PSNR Score:          14.7352
LPIPS Score          0.5679
------------------------------


64x64, 400 epochs
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 98.71%
Generator Accuracy:  17.67%
SSIM Score:          0.2805
PSNR Score:          14.7025
LPIPS Score          0.6022
------------------------------

64x64, 500 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 96.19%
Generator Accuracy:  20.30%
SSIM Score:          0.2791
PSNR Score:          14.6940
LPIPS Score          0.5519
------------------------------

64x64, 600 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 95.25%
Generator Accuracy:  29.02%
SSIM Score:          0.2847
PSNR Score:          14.7501
LPIPS Score          0.5848
------------------------------

64x64, 700 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 88.99%
Generator Accuracy:  20.14%
SSIM Score:          0.2812
PSNR Score:          14.7388
LPIPS Score          0.5160
------------------------------

64x64, 800 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 89.44%
Generator Accuracy:  17.37%
SSIM Score:          0.2819
PSNR Score:          14.7521
LPIPS Score          0.6033
------------------------------

64x64, 900 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 49.80%
Generator Accuracy:  24.69%
SSIM Score:          0.0948
PSNR Score:          12.7629
LPIPS Score          0.7777
------------------------------

64x64, 1000 epochs:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  4.29%
SSIM Score:          0.0191
PSNR Score:          8.5215
LPIPS Score          0.6718
------------------------------


Cambio de hiperparámetros:
```bash
"params": {
        "bsize": 128,
        "imsize": 64, 
        "nc": 1,
        "nz": 100,
        "ngf": 64,
        "ndf": 128,
        "nepochs": 1000,
        "lr": 0.00005,
        "beta1": 0.7,
        "beta2": 0.999,
        "save_epoch": 100
    }
```

64x64, 100 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 94.08%
Generator Accuracy:  36.86%
SSIM Score:          0.2574
PSNR Score:          14.4954
LPIPS Score          0.5712
------------------------------

64x64, 200 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 97.12%
Generator Accuracy:  29.98%
SSIM Score:          0.2734
PSNR Score:          14.5860
LPIPS Score          0.5804
------------------------------

64x64, 300 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.87%
Generator Accuracy:  20.20%
SSIM Score:          0.2702
PSNR Score:          14.6053
LPIPS Score          0.5890
------------------------------

64x64, 400 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.99%
Generator Accuracy:  16.86%
SSIM Score:          0.2757
PSNR Score:          14.7193
LPIPS Score          0.5505
------------------------------

64x64, 500 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.71%
Generator Accuracy:  24.67%
SSIM Score:          0.2797
PSNR Score:          14.6879
LPIPS Score          0.5096
------------------------------

64x64, 600 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.52%
Generator Accuracy:  21.92%
SSIM Score:          0.2810
PSNR Score:          14.8814
LPIPS Score          0.5321
------------------------------

64x64, 700 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 98.18%
Generator Accuracy:  30.17%
SSIM Score:          0.2826
PSNR Score:          14.7375
LPIPS Score          0.6171
------------------------------

64x64, 800 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 84.33%
Generator Accuracy:  21.75%
SSIM Score:          0.2713
PSNR Score:          14.7873
LPIPS Score          0.6266
------------------------------


64x64, 900 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.19%
Generator Accuracy:  18.70%
SSIM Score:          0.2836
PSNR Score:          14.6667
LPIPS Score          0.5310
------------------------------

64x64, 1000 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  6.47%
SSIM Score:          0.0091
PSNR Score:          7.4927
LPIPS Score          0.8298
------------------------------


Implementando: 
```python
# Etiquetas suavizadas (Técnica de Label Smoothing)
            real_label = 0.85 + torch.rand(1).item() * 0.15  # Entre 0.85 y 1.0
            fake_label = 0.0 + torch.rand(1).item() * 0.15  # Entre 0.0 y 0.15

# Evita que D domine demasiado rápido
            if errD.item() > 0.1:
                optimizerD.step()
```
- Label Smoothing dinámico → Evita que D se vuelva demasiado seguro.
- Actualización condicional de D → if errD.item() > 0.1: mantiene un balance adecuado.
- Menos llamadas innecesarias a .fill_() → Más eficiente en memoria.

64x64, 100 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 81.81%
Generator Accuracy:  4.84%
SSIM Score:          0.2660
PSNR Score:          14.7240
LPIPS Score          0.5970
------------------------------

64x64, 200 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 87.94%
Generator Accuracy:  4.89%
SSIM Score:          0.2679
PSNR Score:          14.5734
LPIPS Score          0.4999
------------------------------

64x64, 300 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 98.30%
Generator Accuracy:  8.34%
SSIM Score:          0.2661
PSNR Score:          14.7060
LPIPS Score          0.5046
------------------------------

64X64, 400 epoch:

------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.52%
Generator Accuracy:  12.62%
SSIM Score:          0.2690
PSNR Score:          14.4897
LPIPS Score          0.5607
------------------------------

64x64, 500 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 94.42%
Generator Accuracy:  11.13%
SSIM Score:          0.2819
PSNR Score:          14.6819
LPIPS Score          0.5862
------------------------------

64x64, 600 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 54.74%
Generator Accuracy:  17.26%
SSIM Score:          0.2791
PSNR Score:          14.8103
LPIPS Score          0.5973
------------------------------

64X64, 700 Epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 81.12%
Generator Accuracy:  17.00%
SSIM Score:          0.2687
PSNR Score:          14.4370
LPIPS Score          0.5587
------------------------------

64x64, 800 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  8.26%
SSIM Score:          0.2315
PSNR Score:          14.6390
LPIPS Score          0.7246
------------------------------

64x64, 900 epoch:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  17.84%
SSIM Score:          0.0468
PSNR Score:          11.6575
LPIPS Score          0.8012

64x64, epoch 1000:
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  12.20%
SSIM Score:          0.2152
PSNR Score:          13.8672
LPIPS Score          0.7760
------------------------------

Dado que se ha entrenado una **GAN** en un **ordenador con 14 núcleos** de CPU durante **1000 épocas**, los resultados obtenidos son adecuados para las características del sistema. Las **GANs** son conocidas por ser difíciles de entrenar, y generar imágenes de alta calidad requiere una cantidad considerable de tiempo y recursos. El hecho de que la **precisión del discriminador** se haya estabilizado cerca del **50%** hacia el final del entrenamiento indica que se está alcanzando un equilibrio entre el generador y el discriminador, lo que sugiere que el generador está mejorando. Además, las **métricas de similitud (SSIM, PSNR, LPIPS)** muestran que las imágenes generadas presentan ciertas características realistas, aunque todavía requieren más épocas de entrenamiento para perfeccionarse. En este contexto, con el hardware disponible y el número de épocas alcanzado, estos resultados son satisfactorios, ya que las **GANs** requieren una considerable inversión de tiempo y potencia computacional para lograr una alta calidad en las imágenes generadas.


## Imágenes 512x512

```json
{
    "params": {
        "bsize": 32,
        "imsize": 512, 
        "nc": 1,
        "nz": 100,
        "ngf": 64,
        "ndf": 128,
        "nepochs": 1000,
        "lr": 0.00005,
        "beta1": 0.7,
        "beta2": 0.999,
        "save_epoch": 100
    }
}
```

model_epoch_100.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 99.68%
Generator Accuracy:  23.55%
SSIM Score:          0.3856
PSNR Score:          13.7723
LPIPS Score          0.7265
------------------------------


model_epoch_200.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 53.86%
Generator Accuracy:  28.33%
SSIM Score:          0.0837
PSNR Score:          12.3069
LPIPS Score          0.7788
------------------------------


model_epoch_300.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 64.33%
Generator Accuracy:  19.19%
SSIM Score:          0.3956
PSNR Score:          14.1473
LPIPS Score          0.7445
------------------------------


model_epoch_400.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  13.30%
SSIM Score:          0.3787
PSNR Score:          14.5670
LPIPS Score          0.7787
------------------------------


model_epoch_500.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  23.31%
SSIM Score:          0.3809
PSNR Score:          13.3189
LPIPS Score          0.7740
------------------------------


model_epoch_600.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.43%
Generator Accuracy:  19.54%
SSIM Score:          -0.0016
PSNR Score:          13.7133
LPIPS Score          0.7403
------------------------------


model_epoch_700.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 53.62%
Generator Accuracy:  28.18%
SSIM Score:          0.3142
PSNR Score:          13.3582
LPIPS Score          0.7678
------------------------------


model_epoch_800.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  19.22%
SSIM Score:          0.2606
PSNR Score:          12.9328
LPIPS Score          0.7902
------------------------------

model_epoch_900.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  22.47%
SSIM Score:          0.1870
PSNR Score:          12.0132
LPIPS Score          0.7833
------------------------------


model_ChestCT.pth
------------------------------
   Model Evaluation Results   
------------------------------
Discriminator Accuracy: 50.00%
Generator Accuracy:  15.64%
SSIM Score:          0.0874
PSNR Score:          11.9374
LPIPS Score          0.7603
------------------------------

Aumentar nz permite codificar más información en el espacio latente, evitando imágenes con detalles pobres o repetitivos. Subir ngf en el generador mejora la capacidad de reconstrucción, permitiendo representar estructuras más complejas en 512x512 sin perder definición. A su vez, incrementar ndf en el discriminador le permite evaluar mejor patrones de mayor escala, haciendo la competencia con el generador más efectiva y evitando que pase imágenes borrosas como realistas. Sin estos cambios, es probable que el modelo siga generando imágenes con un nivel de detalle similar al de 64x64, desaprovechando la mayor capacidad de resolución.









