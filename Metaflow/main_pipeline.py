from metaflow import FlowSpec, step, Parameter
import pandas as pd
import random
from ..Data.generate_data import process_dicom_folders

"""
python ChestCancerGAN.py run 

"""

class ChestGAN(FlowSpec):

    @step
    def start(self):
        """ Selección de Imágenes para el modelo """
        # generate-data.py
        # Cogemos las imágenes de TCIA y las dividimos en "Data-Transformed" y "Data-Discarded" (S3 Bucket divido en carpetas)
        process_dicom_folders(
            path_NBIA_Data='../../../../ChestCT-NBIA/manifest-1608669183333',  # Path to the folder containing metadata.csv and Lung-PET-CT-Dx
            reference_images_paths=['Imagen_Ref1.png', 'Imagen_Ref2.png', 'Imagen_Ref3.png'],
            transformed_dir='Data-Transformed/cancer',
            discarded_dir='Discarded/',
            threshold=0.3500
        )

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
        # Si cae más de un umbral., no generar img y que se le envíe al admin , generar report entrenamiento fallido

    @step
    def generate_report(self):
        """ Generar un informe mensual """
        # report.py --> report_{fecha}.pdf en una carpeta del S3 Bucket
        # Genera un informe en PDF con las métricas de evaluación y la gráfica de pérdidas del generador y discriminador
        # Desde la Web los administradores deberían poder acceder a estos PDFs

    # SÍ PASA LA EVALUACIÓN
    @step
    def generate_imgs(self):
        """ Generar Imágenes Sintéticas """
        # generate.py --> Guardar img en "Data-Transformed" como Sinthetic_X_{fecha}.png siendo X el número de img generada. 
        # Aquí desde la web se podrá acceder a las img sintéticas y presentarlas a usuarios


    @step
    def end(self):
        """Fin del pipeline."""
        print("Pipeline finalizado.")

if __name__ == "__main__":
    ChestGAN()