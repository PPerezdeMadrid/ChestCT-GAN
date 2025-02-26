import boto3
import os
import json
from botocore.exceptions import NoCredentialsError

"""
Todo: revisar credenciales de aws. roles IAM.
"""
s3_client = boto3.client('s3')

with open('GAN_PyTorch/config.json', 'r') as json_file:
    config = json.load(json_file)


def upload_files_to_s3(local_directory, bucket_name, folder_name, file_extension=None):
    try:
        s3_client.list_buckets()  # Intentar listar buckets para validar credenciales
    except NoCredentialsError:
        return "⚠️ Credenciales de AWS no encontradas. Verifica tu configuración."
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            # Verificar si el archivo tiene la extensión especificada
            if file_extension is None or file.lower().endswith(file_extension):
                local_file_path = os.path.join(root, file)
                s3_file_path = f"{folder_name}/{file}"

                try:
                    s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
                    return f"Archivo {file} cargado a {s3_file_path}"
                except FileNotFoundError:
                    return f"El archivo {file} no se encontró."
                except NoCredentialsError:
                    return "Credenciales de AWS no encontradas."
                except Exception as e:
                    return f"Error al cargar {file}: {e}"


local_directory_images = config["model"]["image_path_dcgan"]
local_directory_evaluation = config["model"]["evaluation_dcgan"] 

bucket_name = 'tfg-chestgan-bucket'

folder_name_images = 'images_dcgan'
folder_name_evaluation = 'evaluation_dcgan'

# upload_files_to_s3(local_directory_images, bucket_name, folder_name_images, '.png')

# upload_files_to_s3(local_directory_evaluation, bucket_name, folder_name_evaluation)
upload_files_to_s3("evaluation/evaluation_dcgan", bucket_name, folder_name_evaluation)
