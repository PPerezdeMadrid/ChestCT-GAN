from metaflow import FlowSpec, step, Parameter
import pandas as pd
import json, datetime
from Data.generateData import process_dicom_folders
from GAN_PyTorch import train_pipeline, eval_model_pipeline, generate_pipeline, report_pipeline, optimize_pipeline
from metaflow.plugins import kubernetes
from upload_s3Bucket import upload_files_to_s3

"""
python ChestCancerGAN.py run|show|check
"""

def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)


class ChestGAN(FlowSpec):

    config = load_config()
    dataset_nbia_path = config["datasets"]["nbia"]
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Parámetros
    model_type = Parameter('model_type', default='dcgan', help='Modelo a entrenar: dcgan o wgan')
    dataset = Parameter('dataset', default='nbia', help='Dataset: chestct o nbia')
    num_output = Parameter('num_output', default=100, help='Number of images to be generated')
    

    @step
    def start(self):
        """ Selección de Imágenes para el modelo """

        print("\033[94mChoosing Data...\033[0m")
        process_dicom_folders(
            path_NBIA_Data= self.dataset_nbia_path,
            reference_images_paths=['Data/Imagen_Ref1.png', 'Data/Imagen_Ref2.png', 'Data/Imagen_Ref3.png'],
            transformed_dir='Data/Data-Transformed/cancer',
            discarded_dir='Data/Data-Discarded/',
            threshold=0.3500
        )
        self.next(self.train_model) 

    # @kubernetes(cpu=4, memory=16)
    @step
    def train_model(self):
        """ Entrenar el modelo """

        print("\033[94mTraining model...\033[0m")
        arg = {
            'model_type': self.model_type,
            'dataset': self.dataset,
        }
        self.finalmodel_name, self.plot_path, self.csv_log = train_pipeline.main(arg, self.config["params"] ) 
        # self.finalmodel_name = "model_ChestCT_2025-02-17.pth"
        # self.plot_path = "evaluation/evaluation_dcgan/training_losses_2025-02-17_20-14-53_dcgan.png"
        # self.csv_log = "evaluation/training_log_dcgan_2025-02-17.csv"
        self.next(self.eval_model)
        # evaluation/evaluation_{model}/training_log_{model}_{fecha}.csv ==> Logs de cada epoch
        # evaluation/evaluation_{model}/training_losses_{current_time}_{model}.png ==> Pérdida del G y D
        # model/model_{model}/model_ChestCT.pth

    @step
    # @kubernetes(cpu=2, memory=16)
    def eval_model(self):
        """ Evaluar el modelo """
        print("\033[94mEvaluating model...\033[0m")
        # 1) Generamos img_eval_lpips.png para poder aplicar la métrica LPISP ==> ../Data/images/images_{model_type}/img_eval_lpips.png'
        generate_pipeline.generate_one_img(model_type= self.model_type, img_name="img_eval_lpips.png", model_name=self.finalmodel_name)
        # 2) Ejecutamos eval model ==> evaluation/evaluation_{model}/EvalModel_{model_type}_{date}.md
        accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score, self.eval_md_path = eval_model_pipeline.main(self.model_type, self.dataset)
        self.model_score = eval_model_pipeline.validate_eval(accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score)

        print(f'\033[94mEvaluation Score ==> {self.model_score}\033[0m')
        self.next(self.generate_imgs)

    # @kubernetes(cpu=2, memory=8)
    @step
    def generate_imgs(self):
        """ Generar Imágenes Sintéticas """
        if self.model_score>7:
            print("\033[94mGenerating Images...\033[0m")
            # Generar imágenes en images/images_{model}/Synthetic_{model_type}_{i + 1}_{current_date}.png
            generate_pipeline.main(self.model_type, self.num_output, self.finalmodel_name)

        else:
            print("\033[94mImages are not going to be generated due to a bad scoring in the evaluation step.\033[0m")
            print("\033[1;34m[INFO] \033[94mInitializing hyperparameter optimization process...\033[0m")
            optimize_pipeline.main(f"BestParameters_{self.current_date}", f"../evaluation/evaluation_{self.model_type}", self.model_type, self.dataset)

        self.next(self.generate_report)

    @step
    # @kubernetes(cpu=1, memory=4)
    def generate_report(self):
        """ Generar un informe mensual """
        print("\033[94mCreating a report...\033[0m")

        model_trained = f"model/model_{self.model_type}/{self.finalmodel_name}"
        filename = report_pipeline.generate_report_pdf(
            data_transformed = "../Data/Data-Transformed/cancer/",
            data_discarded="../Data/Data-Discarded",
            model=self.model_type,
            model_trained=model_trained,
            log_training=self.csv_log,
            graph_training=self.plot_path,
            img_to_eval=f"../Data/images/images_{self.model_type}/img_eval_lpips.png'",
            report_eval=self.eval_md_path,
            image_path=f"../images/images_{self.model_type}",
            filename=f"report_{self.current_date}.pdf"
        )
        # report.py --> report_{fecha}.pdf en una carpeta del S3 Bucket
        # Genera un informe en PDF con las métricas de evaluación y la gráfica de pérdidas del generador y discriminador
        print(f"\033[94mReport created at {filename}\033[0m")
        self.next(self.upload_files_cloud)

    @step 
    def upload_files_cloud(self):
        config = load_config()
        bucket_name = 'tfg-chestgan-bucket'

        folder_name_images = f"images_{self.model_type}"
        folder_name_evaluation = f"evaluation_{self.model_type}"

        
        if self.model_type == 'dcgan':
            # Subir imágenes generadas por el modelo
            img_gen_upload = upload_files_to_s3(config["model"]["image_path_dcgan"], bucket_name, folder_name_images, '.png')
            # Subir archivos de evaluación
            eval_upload = upload_files_to_s3(config["model"]["evaluation_dcgan"], bucket_name, folder_name_evaluation)
        elif self.model_type == 'wgan':
            img_gen_upload = upload_files_to_s3(config["model"]["image_path_wgan"], bucket_name, folder_name_images, '.png')
            eval_upload = upload_files_to_s3(config["model"]["evaluation_wgan"], bucket_name, folder_name_evaluation)
        else:
            print("\033[94mThe model type is not recognized, no images and documents will be uploaded to the s3 bucket. \033[0m")

        print(img_gen_upload)
        print(eval_upload)
        self.next(self.end)

    @step
    def end(self):
        """Fin del pipeline."""
        print("\033[94mThe pipeline has come to an END\033[0m")

if __name__ == "__main__":
    ChestGAN()


    """
Nota: añadir @cards En Metaflow, la etiqueta @card se usa para generar visualizaciones y reportes en la interfaz de Metaflow UI.
 Permite adjuntar información en formato Markdown, HTML o JSON a un paso específico del flujo (Step), facilitando la inspección de r
 esultados, métricas o gráficos directamente en la UI.
"""
