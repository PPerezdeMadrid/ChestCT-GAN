1. Inception Score (IS)
El Inception Score es una métrica popular para evaluar la calidad de las imágenes generadas. Esta métrica utiliza un modelo de clasificación preentrenado (Inception v3) para evaluar la diversidad y la calidad de las imágenes. PARA IMÁGENES EN BLANCO Y NEGRO NO ES LO IDEAL.

2. Fréchet Inception Distance (FID)
El FID es otra métrica que compara la distribución de características de las imágenes reales y generadas. Un valor más bajo indica que las imágenes generadas son más similares a las reales. NO HE CONSEGUIDO QUE FUNCIONE POR EL TEMA DE LOS CANALES.

3. Porcentaje de acierto del Generador

4. Structural Similarity Index (SSIM)
El SSIM es una métrica popular para evaluar la calidad de las imágenes, especialmente cuando se comparan imágenes generadas con imágenes reales. Este índice compara la luminancia, el contraste y la estructura de las imágenes.

Ventaja: SSIM tiene en cuenta características perceptuales, como la estructura y la textura, lo que lo hace útil para la evaluación de imágenes médicas como las radiografías.

gepeto dice:
- SSIM y PSNR son métricas ampliamente utilizadas en la evaluación de imágenes médicas y te proporcionarán una idea clara de la calidad visual de las imágenes generadas.
- MSE es útil para obtener una medida numérica, pero puede no reflejar la calidad perceptual.
- DSC es excelente si estás interesado en la segmentación o en la similitud de regiones específicas dentro de las radiografías.
- TV Loss y KL Divergence pueden ser útiles para medir la calidad y suavidad global, respectivamente.