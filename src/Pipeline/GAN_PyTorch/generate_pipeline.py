import torch, json, os
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from datetime import datetime

# Importar los generadores de dcgan y wgan
from GAN_PyTorch.dcgan import Generator as DCGANGenerator
from GAN_PyTorch.wgan import Generator as WGANGenerator

def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        config = json.load(json_file)
    return config

def get_device():
    """Determina el dispositivo adecuado para la ejecución."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # MacOS ARM
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Usar GPU CUDA
    elif torch.xpu.is_available():
        return torch.device("xpu")  # Usar XPU (Intel gráficos "Xe")
    else:
        return torch.device("cpu")  # Usar CPU

def load_model(params, model_type, device):
    """Carga el modelo adecuado en función del tipo especificado."""
    print(f"Model type received: {model_type}")
    if model_type == 'dcgan':
        netG = DCGANGenerator(params["params"]).to(device)
        image_path = params["model"]["image_path_dcgan"]
    elif model_type == 'wgan':
        netG = WGANGenerator(params["params"]).to(device)
        image_path = params["model"]["image_path_wgan"]
    else:
        raise ValueError("Unsupported model type")
    
    return netG, image_path

def generate_and_save_images(params, model_type, num_output, load_path, eval_path):
    """Generar y guardar imágenes sintéticas usando el generador del modelo"""

    device = get_device()
    config = load_config()

    # Cargar el modelo
    state_dict = torch.load(load_path, map_location=device)

    # Crear la red generadora según el modelo elegido
    if model_type == 'dcgan':
        image_path = config["model"]["image_path_dcgan"]
        netG = DCGANGenerator(params).to(device)
    elif model_type == 'wgan':
        image_path = config["model"]["image_path_wgan"]
        netG = WGANGenerator(params).to(device)
    else:
        raise ValueError("Unsupported model type")

    # Cargar el modelo previamente entrenado
    netG.load_state_dict(state_dict['generator'])
    print(netG)
    print(num_output)

    # Obtener el vector latente Z de una distribución normal estándar
    noise = torch.randn(num_output, params['nz'], 1, 1, device=device)

    # Desactivar el cálculo de gradientes para acelerar el proceso (No necesitamos actualizar los param del modelo)
    with torch.no_grad():
        # Obtener la imagen generada desde el vector de ruido usando
        # el generador entrenado
        generated_img = netG(noise).detach().cpu()

    # Guardar las imágenes generadas
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    normalized_images = (generated_img + 1) / 2  # normaliza las imágenes a [0, 1]

    # Guardar las imágenes generadas
    image_files = []
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, img in enumerate(normalized_images):
        file_path = os.path.join(image_path, f'Synthetic_{model_type}_{i + 1}_{current_date}.png')
        save_image(img, file_path)
        image_files.append(file_path)
        print(f'==> Imagen guardada en {file_path}')
    
    create_image_grid(image_files, eval_path)


def create_image_grid(image_files, output_path, grid_size=(3, 3)):
    """Crear una cuadrícula de imágenes a partir de los archivos de imagen generados"""

    # Cargar las imágenes usando PIL
    images = [Image.open(file) for file in image_files]

    # Verificar que haya suficientes imágenes
    if len(images) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Se necesitan {grid_size[0] * grid_size[1]} imágenes, pero se recibieron {len(images)}")

    # Crear una imagen en blanco para la cuadrícula
    width, height = images[0].size
    grid_image = Image.new('RGB', (width * grid_size[0], height * grid_size[1]))

    # Colocar las imágenes en la cuadrícula
    for i, img in enumerate(images):
        row = i // grid_size[0]
        col = i % grid_size[0]
        grid_image.paste(img, (col * width, row * height))

    # Guardar la imagen de la cuadrícula
    grid_image_path = os.path.join(output_path, 'image_grid.png')
    grid_image.save(grid_image_path)
    print(f'==> Cuadrícula de imágenes guardada en {grid_image_path}')
   

def generate_one_img(model_type='dcgan', img_name="img_eval_lpips.png", model_name="ChestGAN.pth"):
    """Generar y guardar una sola imagen utilizando el generador del modelo"""
    
    device = get_device()
    config = load_config()
    params = config["params"]

    if model_type=='dcgan':
        model_path = config["model"]["path_dcgan"]
        eval_path = config["model"]["evaluation_dcgan"]
    elif model_type == 'wgan':
        model_path = config["model"]["path_wgan"]
        eval_path = config["model"]["evaluation_wgan"]

    print(f"======> {model_path}/{model_name}")
    state_dict = torch.load(f"{model_path}/{model_name}", map_location=device)

    netG, img_path = load_model(config, model_type, device)

    netG.load_state_dict(state_dict['generator'])
    print(f"Cargado el modelo {model_type} desde {f'{model_path}/{model_name}'}")

    noise = torch.randn(1, params['nz'], 1, 1, device=device)  # Solo 1 imagen

    # Desactivar el cálculo de gradientes para acelerar el proceso
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()

    normalized_img = (generated_img + 1) / 2

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # Guardar la imagen generada
    image_path = os.path.join(eval_path, img_name)
    save_image(normalized_img, image_path)
    print(f'==> Imagen generada y guardada en {image_path}')

    return image_path  


def main(model_type, num_output, model_name):
    num_output = 64  # número de imágenes a generar
    params = load_config("GAN_PyTorch/config.json")

    if model_type=='dcgan':
        model_path = params["model"]["path_dcgan"]
        eval_path = params["model"]["evaluation_dcgan"]
    elif model_type == 'wgan':
        model_path = params["model"]["path_wgan"]
        eval_path = params["model"]["evaluation_wgan"]
    
    load_path = f'{model_path}/{model_name}'

    generate_and_save_images(params, model_type, num_output, load_path, eval_path)

