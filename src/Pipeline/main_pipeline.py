from metaflow import FlowSpec, step, Parameter, card
import json, datetime, requests, os
from Data.generateData import process_dicom_folders, adjust_brightness_in_directory
from GAN_PyTorch import train_pipeline, eval_model_pipeline, generate_pipeline, report_pipeline, optimize_pipeline
from upload_s3Bucket import upload_files_to_s3
import logging

"""
Actualmente, este pipeline está diseñado para usar la arquitectura de DCGAN. 
python ChestCancerGAN.py run|show|check
"""


"""
Para ver los logs en tiempo real, puedes usar el siguiente comando en la terminal:
tail -f pipeline_live.log
"""
logging.basicConfig(
    filename='pipeline_live.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)


class ChestGAN(FlowSpec):

    config = load_config()
    dataset_nbia_path = config["datasets"]["nbia"]
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Parámetros
    model_type = Parameter('model_type', default='dcgan', help='Modelo a entrenar: dcgan o wgan')
    dataset = Parameter('dataset', default='nbia', help='Dataset: chestct o nbia')
    num_output = Parameter('num_output', default=100, help='Number of images to be generated')
    ip_frontend = Parameter('ip_frontend', default="www.chestgan.tech", help='IP Address of the frontend')
    yaml_path = Parameter('yaml_path', default="GAN_PyTorch/weights.yaml", help='Path to the weights yaml for the evaluation weights')
    
    @card
    @step
    def start(self):
        """ Selección de Imágenes para el modelo """
        transformed_dir = 'Data/Data-Transformed/cancer'

        if self.dataset == 'nbia':
            print("\033[94mChecking if data is already prepared...\033[0m")
            already_prepared = os.path.exists(transformed_dir) and len(os.listdir(transformed_dir)) > 0

            if not already_prepared:
            # Si has elegido el dataset "nbia"
                if self.dataset == 'nbia':
                    print("\033[94mChoosing Data...\033[0m")
                    process_dicom_folders(
                        metadata_csv_path='Data/manifest-160866918333/metadata.csv',  # Ruta directa al CSV
                        dicom_root_dir=self.dataset_nbia_path,  # Carpeta que contiene las subcarpetas con DICOMs, definida en config.json
                        reference_images_paths=['Data/Img_ref/Imagen_Ref1.png', 'Data/Img_ref/Imagen_Ref2.png', 'Data/Img_ref/Imagen_Ref3.png', 'Data/Img_ref/Imagen_Ref4.png', 'Data/Img_ref/Imagen_Ref5.png',],
                        transformed_dir=transformed_dir,
                        discarded_dir='Data/Discarded/',
                        threshold=0.3100
                    )

                    adjust_brightness_in_directory('Data-Transformed/cancer',target_brightness=28)

                    """ Ejemplo de uso del dataset reducido de TCIA"""
                    # process_dicom_folders(
                    # path_NBIA_Data=self.dataset_nbia_path,
                    #     reference_images_paths=[
                    #         'Data/Img_ref/Imagen_Ref1.png',
                    #         'Data/Img_ref/Imagen_Ref2.png',
                    #         'Data/Img_ref/Imagen_Ref3.png',
                    #         'Data/Img_ref/Imagen_Ref4.png',
                    #         'Data/Img_ref/Imagen_Ref5.png',
                    #         'Data/Img_ref/Imagen_Ref6.png',
                    #         'Data/Img_ref/Imagen_Ref7.png',
                    #         'Data/Img_ref/Imagen_Ref8.png',
                    #         'Data/Img_ref/Imagen_Ref9.png',
                    #         'Data/Img_ref/Imagen_Ref10.png',
                    #         'Data/Img_ref/Imagen_Ref11.png',
                    #     ],
                    #     discarded_reference_images_paths=[
                    #         'Data/Img_ref/Imagen_Discarded_1.png',
                    #         'Data/Img_ref/Imagen_Discarded_2.png',
                    #         'Data/Img_ref/Imagen_Discarded_3.png',
                    #         'Data/Img_ref/Imagen_Discarded_4.png',
                    #         'Data/Img_ref/Imagen_Discarded_5.png'
                    #     ],  
                    #     transformed_dir='Data/Data-Transformed/cancer',
                    #     discarded_dir='Data/Data-Discarded/',
                    #     threshold_lpips=0.3500,
                    #     threshold_psnr=20.0,
                    #     threshold_discard_lpips=0.3500,
                    #     threshold_discard_psnr=20.0
                    # )

                else:
                    print("\033[94mData selected do not need preprocessing\033[0m")
            else:
                print("\033[94mData already prepared, skipping preprocessing step.\033[0m")
                
        self.next(self.train_model)
        

    # @kubernetes(cpu=4, memory=16)
    @card
    @step
    def train_model(self):
        """ Entrenar el modelo """

        print("\033[94mTraining model...\033[0m")
        arg = {
                'model_type': self.model_type,
                'dataset': self.dataset,
        }

        self.finalmodel_name, self.plot_path, self.csv_log = train_pipeline.main(arg, self.config["params"] ) 
            
        self.next(self.eval_model)
        # evaluation/evaluation_{model}/training_log_{model}_{fecha}.csv ==> Logs de cada epoch
        # evaluation/evaluation_{model}/training_losses_{current_time}_{model}.png ==> Pérdida del G y D
        # model/model_{model}/model_ChestCT.pth

    @card
    @step
    # @kubernetes(cpu=2, memory=16)
    def eval_model(self):
        """ Evaluar el modelo """
        print("\033[94mEvaluating model...\033[0m")
        # 1) Generamos img_eval_lpips.png para poder aplicar la métrica LPISP ==> ../Data/images/images_{model_type}/img_eval_lpips.png'
        generate_pipeline.generate_one_img(model_type= self.model_type, img_name="img_eval_lpips.png", model_name=self.finalmodel_name)
        # 2) Ejecutamos eval model ==> evaluation/evaluation_{model}/EvalModel_{model_type}_{date}.md
        print(f"Evaluating model of type: {self.model_type} on dataset: {self.dataset} with final model path: {self.finalmodel_name}")
        img_ref_path = f"Data/Img_ref/Imagen_Ref1.png"
        self.accuracy_discriminator, self.accuracy_generator, self.ssim_score, self.psnr_score, self.lpips_score, self.eval_md_path = eval_model_pipeline.main(self.model_type, self.dataset, self.finalmodel_name, img_ref_path)
        self.model_score = eval_model_pipeline.validate_eval(self.accuracy_discriminator, self.accuracy_generator, self.ssim_score, self.psnr_score, self.lpips_score, self.yaml_path)

        print(f'\033[94mEvaluation Score ==> {self.model_score}\033[0m')
        self.next(self.generate_imgs)

    # @kubernetes(cpu=2, memory=8)
    @card
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
            img_ref_path = f"Data/Img_ref/Imagen_Ref1.png"
            optimize_pipeline.main(
                f"BestParameters_{self.current_date}", 
                self.config["model"][f'evaluation_{self.model_type}'], 
                self.model_type, self.dataset, 
                self.finalmodel_name, 
                img_ref_path, 
                n_trials=5 # Puedes ajustar el número de intentos según tus necesidades
                ) 

        self.next(self.generate_report)

    @card
    @step
    # @kubernetes(cpu=1, memory=4)
    def generate_report(self):
        """ Generar un informe mensual """
        print("\033[94mCreating a report...\033[0m")

        model_trained = f"model/model_{self.model_type}/{self.finalmodel_name}"
        if self.model_type == 'dcgan':
            report = report_pipeline.generate_report_pdf(
                data_transformed = "Data/Data-Transformed/cancer/",
                data_discarded="Data/Data-Discarded",
                model=self.model_type,
                model_trained=model_trained,
                log_training=self.csv_log,
                graph_training=self.plot_path,
                img_to_eval=f"Data/images/images_{self.model_type}/img_eval_lpips.png'",
                report_eval=self.eval_md_path,
                image_path=self.config["model"][f"image_path_{self.model_type}"],
                filename=f"report_{self.current_date}.pdf",
                accuracy_discriminator=self.accuracy_discriminator,
                accuracy_generator=self.accuracy_generator,
                ssim_score=self.ssim_score,
                psnr_score=self.psnr_score,
                lpips_score=self.lpips_score
            )
        else:
            report = report_pipeline.generate_report_pdf(
                data_transformed = "Data/Data-Transformed/cancer/",
                data_discarded="Data/Data-Discarded",
                model=self.model_type,
                model_trained=model_trained,
                log_training=self.csv_log,
                graph_training=self.plot_path,
                img_to_eval=f"Data/images/images_{self.model_type}/img_eval_lpips.png'",
                report_eval=self.eval_md_path,
                image_path=self.config["model"][f"image_path_{self.model_type}"],
                filename=f"report_{self.current_date}.pdf",
                accuracy_discriminator=0.0,  # Placeholder for WGAN, as it does not use discriminator accuracy
                accuracy_generator=0.0,  # Placeholder for WGAN, as it does not use generator accuracy
                ssim_score=self.ssim_score,
                psnr_score=self.psnr_score,
                lpips_score=self.lpips_score
            )
        # report.py --> report_{fecha}.pdf en una carpeta del S3 Bucket
        # Genera un informe en PDF con las métricas de evaluación y la gráfica de pérdidas del generador y discriminador
        print(f"\033[94mReport created at {report}\033[0m")
        self.next(self.upload_files_cloud)
   
    @card
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

        self.next(self.end)
   
    @card
    @step
    def end(self):
        """Fin del pipeline."""
        # Notificar al frontend con HTTPS primero, luego intentar con HTTP si falla
        for protocol in ["https", "http"]:
            webhook_url = f"{protocol}://{self.ip_frontend}/notify"
            try:
                response = requests.post(webhook_url, json={
                    "mensaje": f"Se ha ejecutado el pipeline de MLOps a fecha {self.current_date}"
                }, timeout=5)
                print(f"Notificación enviada con {protocol.upper()}")
                print("Respuesta del servidor:", response.json())
                break  # Salir del bucle si la notificación fue exitosa
            except Exception as e:
                print(f"Error notificando al frontend con {protocol.upper()}: {e}")
        print("\033[94mThe pipeline has come to an END\033[0m")

if __name__ == "__main__":
    ChestGAN()


