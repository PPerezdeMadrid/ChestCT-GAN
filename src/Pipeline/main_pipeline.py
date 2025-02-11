from metaflow import FlowSpec, step, Parameter
import pandas as pd
import random
from Data.generateData import process_dicom_folders
from GAN_PyTorch import train_pipeline # type: ignore

"""
python ChestCancerGAN.py run|show|check
"""
dataset_path = "../../../../ChestCT-NBIA/manifest-1608669183333" # CAMBIAR !!


class ChestGAN(FlowSpec):

    # Parámetros
    model_type = Parameter('model_type', default='dcgan', help='Modelo a entrenar: dcgan o wgan')
    dataset = Parameter('dataset', default='nbia', help='Dataset: chestct o nbia')
    model_path = Parameter('model_path', help='Ruta para guardar el modelo', default='models/')
    

    @step
    def start(self):
        """ Selección de Imágenes para el modelo """
        # generate-data.py
        # Cogemos las imágenes de TCIA y las dividimos en "Data-Transformed" y "Data-Discarded" (S3 Bucket divido en carpetas)
        print("\033[94mChoosing Data...\033[0m")
        """
        process_dicom_folders(
            path_NBIA_Data= dataset_path,
            reference_images_paths=['Data/Imagen_Ref1.png', 'Data/Imagen_Ref2.png', 'Data/Imagen_Ref3.png'],
            transformed_dir='Data/Data-Transformed/cancer',
            discarded_dir='Data/Data-Discarded/',
            threshold=0.3500
        )
        """
        self.next(self.train_model) 

    @step
    def train_model(self):
        """ Entrenar el modelo """
        # Utilizar las imágenes de Data-Transformed para el entrenamiento
        # Guardar el modelo "ChestTC_GAN.pth" en un servidor virtual (EC2)

        print("\033[94mTraining model...\033[0m")
        params = {
            'model_type': self.model_type,
            'dataset': self.dataset,
            'model_path': self.model_path
        }
        train_pipeline.main(params) 
        # Se guarda los logs img del modelo de las pérdidas del generador y discriminador r
        self.next(self.eval_model)

    @step
    def eval_model(self):
        """ Evaluar el modelo """
        # eval_model.py --> archivo EvalModel_{fecha}.md
        # graphLog.py --> Guardar img LossDLossG_{fecha}.png
        # Si cae más de un umbral, no generar img y que se le envíe al admin , generar report entrenamiento fallido
        self.passed_evaluation = random.choice([True, False])  # Simulación del resultado
        print("\033[94mEvaluating model...\033[0m")
        print(f'\033[94mEvaluation ==> {self.passed_evaluation}\033[0m')
        # self.next(self.generate_imgs if self.passed_evaluation else self.generate_report) --> Metaflow no permite 
        self.next(self.generate_report)


    @step
    def generate_report(self):
        """ Generar un informe mensual """
        # report.py --> report_{fecha}.pdf en una carpeta del S3 Bucket
        # Genera un informe en PDF con las métricas de evaluación y la gráfica de pérdidas del generador y discriminador
        # Desde la Web los administradores deberían poder acceder a estos PDFs
        print("\033[94mCreating a report...\033[0m")
        self.next(self.generate_imgs)

    # SÍ PASA LA EVALUACIÓN
    @step
    def generate_imgs(self):
        """ Generar Imágenes Sintéticas """
        # generate.py --> Guardar img en "Data-Transformed" como Sinthetic_X_{fecha}.png siendo X el número de img generada. 
        # Aquí desde la web se podrá acceder a las img sintéticas y presentarlas a usuarios
        if self.passed_evaluation:
            print("\033[94mGenerating Images...\033[0m")
        else:
            print("\033[94mImages are not going to be generated due to a bad scoring in the evaluation step.\033[0m")
        self.next(self.end)


    @step
    def end(self):
        """Fin del pipeline."""
        print("\033[94mThe pipeline has come to an END\033[0m")

if __name__ == "__main__":
    ChestGAN().run()