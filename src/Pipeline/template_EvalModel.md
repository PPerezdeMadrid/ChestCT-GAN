# ChestGAN - Informe de Evaluación del Modelo

## Visión General del Modelo
**Tipo de Modelo:** {model_type.upper()}  

---

## Métricas de Desempeño

### Precisión del Discriminador: {accuracy_discriminator * 100:.2f}%
Capacidad del discriminador para distinguir entre imágenes reales y generadas.

### Precisión del Generador: {accuracy_generator * 100:.2f}%
Tasa de éxito del generador en la creación de imágenes médicas sintéticas realistas.

### SSIM (Índice de Similitud Estructural): {ssim_score:.4f}
Mide la calidad perceptual de las imágenes generadas en comparación con las imágenes médicas reales.

### PSNR (Relación Señal-Ruido de Pico): {psnr_score:.2f} dB
Indica la calidad de las imágenes basada en la relación señal-ruido.

### LPIPS (Similitud Perceptual de Parches de Imagen Aprendida): {lpips_score:.4f}
Cuantifica la similitud perceptual entre las imágenes generadas y las reales según la percepción humana.


