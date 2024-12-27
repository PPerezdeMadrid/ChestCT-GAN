import torch
import random
import json
import os
import argparse
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importar los generadores de dcgan y wgan
from dcgan import Generator as DCGANGenerator
from wgan import Generator as WGANGenerator

# Cargar la configuración
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]

# Configuración de los argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default=f'{config["model"]["path_dcgan"]}/model_ChestCT.pth', help='Checkpoint to load path from')  # Modelo previamente entrenado
parser.add_argument('--num_output', default=64, type=int, help='Number of generated outputs')  # Número de salidas generadas
parser.add_argument('--model', choices=['dcgan', 'wgan'], default='dcgan', help='Model architecture to use: dcgan or wgan')  # Elegir el modelo

args = parser.parse_args()

# Configurar el dispositivo de ejecución (GPU o CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")  # MacOS ARM
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Usar GPU CUDA
elif torch.xpu.is_available():
    device = torch.device("xpu")  # Usar XPU (intel gráficos "Xe")
else:
    device = torch.device("cpu")  # Usar CPU 

print(f"Using device: {device}")  


# Cargar el modelo
state_dict = torch.load(args.load_path, map_location=device)
params = state_dict['params']

# Crear la red generadora según el modelo elegido
if args.model == 'dcgan':
    model_path = config["model"]["path_dcgan"]
    image_path = config["model"]["image_path_dcgan"]
    netG = DCGANGenerator(params).to(device)
elif args.model == 'wgan':
    model_path = config["model"]["path_wgan"]
    image_path = config["model"]["image_path_wgan"]
    netG = WGANGenerator(params).to(device)

# Cargar el modelo previamente entrenado
netG.load_state_dict(state_dict['generator'])
print(netG)
print(args.num_output)

# Obtener el vector latente Z de una distribución normal estándar
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

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
for i, img in enumerate(normalized_images):
    file_path = os.path.join(image_path, f'generated_image_{i + 1}.png')
    save_image(img, file_path)
    print(f'==> Imagen guardada en {file_path}')

# Mostrar las imágenes generadas
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))
plt.show()
