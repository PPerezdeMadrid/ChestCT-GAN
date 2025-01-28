# Pipeline de **MLOPS** con **Metaflow**
Este documento presenta una idea de como quedaría el archivo principal de metaflow

```python
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import random
"""
python ChestCancerGAN.py run 

"""

class ChestGAN(FlowSpec):

    csv_file = Parameter("csv_file", help="Path to the movies CSV file")

    @step
    def start(self):
        """ Selección de Imágenes para el modelo """
        # generate-data.py
        # Cogemos las imágenes de TCIA y las dividimos en "Data-Transformed" y "Data-Discarded" (S3 Bucket divido en carpetas)

    @step
    def train_model(self):
        """ Entrenar el modelo """
        # Utilizar las imágenes de Data-Transformed para el entrenamiento
        # Guardar el modelo "ChestTC_GAN.pth" en un servidor virtual (EC2)

    @step
    def eval_model(self):
        """ Evaluar el modelo """
        # eval_model.py --> archivo EvalModel_{fecha}.md
        # graphLog.py --> Guardar img LossDLossG_{fecha}.png

    @step
    def generate_imgs(self):
        """ Generar Imágenes Sintéticas """
        # generate.py --> Guardar img en "Data-Transformed" como Sinthetic_X_{fecha}.png siendo X el número de img generada. 
        # Aquí desde la web se podrá acceder a las img sintéticas y presentarlas a usuarios

    @step
    def generate_report(self):
        """ Generar un informe mensual """
        # report.py --> report_{fecha}.pdf en una carpeta del S3 Bucket
        # Genera un informe en PDF con las métricas de evaluación y la gráfica de pérdidas del generador y discriminador
        # Desde la Web los administradores deberían poder acceder a estos PDFs

    @step
    def end(self):
        """Fin del pipeline."""
        print("Pipeline finalizado.")

if __name__ == "__main__":
    ChestGAN()
```

Preguntas a Raúl:
- ¿Cuántos servidores? (Web, Imágenes, PDF, ...)