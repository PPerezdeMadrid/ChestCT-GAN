# Evaluación del Modelo

```bash
python eval_model.py
```
## 1. Precisión del Discriminador y del Generador
   - Se evalúa la precisión del **Discriminador**, que determina si las imágenes generadas son reales o falsas.
   - Se calcula la **precisión del Generador**, es decir, la capacidad del generador para crear imágenes que engañen al discriminador.
   - La precisión de ambos se calcula y se imprime como porcentaje.


## 2. **SSIM (Structural Similarity Index)**

El **SSIM** (Índice de Similitud Estructural) es una métrica utilizada para evaluar la calidad de las imágenes comparando imágenes generadas con imágenes reales. Esta métrica evalúa tres aspectos fundamentales de la imagen: **luminancia**, **contraste** y **estructura**, lo que la hace particularmente útil cuando se desean evaluar imágenes que contienen detalles importantes, como las imágenes médicas.

El SSIM tiene la ventaja de que no solo mide la diferencia global entre las imágenes, sino que también tiene en cuenta características perceptuales, como la estructura y la textura, lo que lo convierte en una métrica adecuada para imágenes donde los detalles estructurales son cruciales.

### Interpretación de los valores de SSIM:
- **1**: Indica que las dos imágenes son idénticas, es decir, tienen la misma luminancia, contraste y estructura (perfecta similitud).
- **0**: Significa que las imágenes no tienen ninguna similitud estructural.
- **Valores negativos**: Sugerirían que las imágenes son completamente opuestas en términos de luminancia, contraste y estructura.

### Ventajas de SSIM:
- **Perceptual**: SSIM es más cercano a la forma en que los humanos perciben la calidad de las imágenes, ya que evalúa aspectos como la estructura y los detalles finos.
- **Utilidad en imágenes médicas**: Dado que las imágenes médicas contienen detalles complejos y sutiles, SSIM es muy útil para medir la calidad de las imágenes generadas en este contexto.


## 3. Métrica PSNR (Peak Signal-to-Noise Ratio)

El **PSNR** es una métrica utilizada para evaluar la calidad de las imágenes generadas comparándolas con las imágenes originales (reales). Se calcula como la relación entre la máxima potencia posible de una imagen y la potencia del ruido que afecta a la calidad de la imagen. Es especialmente útil para comparar imágenes de referencia con imágenes generadas por modelos como redes generativas.

### Fórmula:
La fórmula del PSNR es la siguiente:
```
PSNR = 10 * log10((MAX_I^2) / MSE)
```
Donde:
- `MAX_I` es el valor máximo posible de la intensidad de los píxeles (para imágenes de 8 bits, es 255).
- `MSE` es el **Mean Squared Error (Error Cuadrático Medio)** entre las imágenes generada y real.

### Interpretación de los valores de PSNR:
- **Valores altos (30 dB - 50 dB)**: Un PSNR alto indica que la imagen generada es de alta calidad y casi indistinguible de la imagen original. En general, valores de PSNR superiores a 30 dB son considerados como de buena calidad para imágenes generadas.
  
- **Valores bajos (menos de 20 dB)**: Los valores bajos de PSNR indican que hay una gran diferencia entre la imagen generada y la original, lo que sugiere una baja calidad en la generación de la imagen. Esto es común cuando el modelo aún no ha aprendido correctamente las características clave de las imágenes.

## 4. LPIPS (Learned Perceptual Image Patch Similarity) 
Es una métrica de evaluación de la calidad de imágenes que mide la similitud perceptual entre dos imágenes. A diferencia de las métricas tradicionales como el Error Cuadrático Medio (MSE) o el Pico de la Relación Señal-Ruido (PSNR), que evalúan la diferencia entre las imágenes a nivel de píxeles, LPIPS se enfoca en cómo las diferencias afectan la percepción humana. Es decir, LPIPS evalúa la calidad de una imagen teniendo en cuenta cómo los humanos perciben esas diferencias visuales.

### ¿Cómo Funciona LPIPS?
- **Extracción de características**: Utiliza redes neuronales preentrenadas (ej. VGG) para extraer características de las imágenes.
- **Cálculo de la distancia perceptual**: Compara las características extraídas de las imágenes en distintas capas de la red.
- **Interpretación**: Una puntuación baja indica mayor similitud perceptual, mientras que una puntuación alta indica mayores diferencias perceptuales.

### Ventajas de LPIPS
- **Más alineado con la percepción humana**: Mide la calidad de las imágenes de acuerdo con cómo los humanos perciben las diferencias visuales.
- **Sensibilidad a características visuales**: Evalúa detalles importantes como bordes, texturas y otros elementos perceptuales relevantes.
- **Aplicaciones en modelos generativos**: Útil para evaluar la calidad de imágenes generadas por modelos como GANs.

### Limitaciones de LPIPS
- **Dependencia de la GPU**: Requiere una GPU para una ejecución eficiente debido a la complejidad computacional de las redes neuronales.
- **Costo computacional**: Más costoso en términos de tiempo de procesamiento comparado con métricas simples como PSNR.
