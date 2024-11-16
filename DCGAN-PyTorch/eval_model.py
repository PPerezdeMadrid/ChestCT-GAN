import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3  #--> versión antigua
from torchvision import models
from torch.nn import functional as F
from dcgan import Generator  # Asegúrate de que este módulo está definido
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from utils import get_chestct 
from dcgan import Generator, Discriminator
import json
from skimage.metrics import structural_similarity as ssim


def print_green(text):
    print("\033[92m" + text + "\033[0m")

print_green("Evaluating model...")

# leer parámetros y modelo:
with open('config.json', 'r') as json_file:
    config = json.load(json_file)


model_path = config["model"]["path"]

print_green("Parameters uploaded")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_green(f'{device} will be used.\n')


# Cargar el modelo entrenado con weights_only=True para una mayor seguridad
checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', weights_only=True)
params = checkpoint['params']

# Get the data
dataloader = get_chestct(params)

# Load the generator and discriminator
netG = Generator(params).to(device)
netD = Discriminator(params).to(device)

netG.load_state_dict(checkpoint['generator'])
netD.load_state_dict(checkpoint['discriminator'])


# Modo evaluación
netG.eval()
netD.eval()


def generate_images(generator, num_images=100, latent_size=100):
    """Genera imágenes usando el generador."""
    noise = torch.randn(num_images, latent_size, 1, 1, device=device)  # Generar ruido
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()

    if fake_images.shape[1] == 1:  # Si solo tiene 1 canal
        fake_images = fake_images.repeat(1, 3, 1, 1)  # Repetir el canal

    return fake_images

def preprocess_images(images, size=(299, 299)):
    """Preprocesa imágenes para Inception-v3."""
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalización a [-1, 1] para un solo canal

    ])
    
    processed_images = []
    for img in images:
        img = transforms.ToPILImage()(img)  # Convertir tensor a PIL Image
        img = preprocess(img)
        processed_images.append(img)
    
    return torch.stack(processed_images)

""""
##############################
    Discriminador y Generador
######################
"""


def evaluate_models(netG, netD, dataloader, device, params):
    # Evaluar el discriminador
    correct_discriminator = 0
    total = 0
    correct_generator = 0

    with torch.no_grad():
        # Evaluar el discriminador
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)

            # Evaluar el discriminador en datos reales
            real_label_tensor = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_data).view(-1)
            correct_discriminator += ((output_real > 0.5).float() == real_label_tensor).sum().item()

            # Generar datos falsos
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)

            # Evaluar el discriminador en datos generados
            fake_label_tensor = torch.full((b_size,), 0, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach()).view(-1)
            correct_discriminator += ((output_fake < 0.5).float() == fake_label_tensor).sum().item()

            total += b_size * 2  # Para datos reales y generados

        # Calcular precisión del discriminador
        accuracy_discriminator = correct_discriminator / total

        # Evaluar la precisión del generador
        for _ in range(100):  # Generar 100 datos de prueba
            noise = torch.randn(1, params['nz'], 1, 1, device=device)
            generated_data = netG(noise)
            output = netD(generated_data.detach()).view(-1)
            if output > 0.5:
                correct_generator += 1

        accuracy_generator = correct_generator / 100

    return accuracy_discriminator, accuracy_generator


accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, device, params)

# Imprimir --> Generador y Discriminador
print(f"{'-' * 30}")
print(f"{'Model Evaluation Results':^30}")
print(f"{'-' * 30}")
print(f"{'Discriminator Accuracy:':<20} {accuracy_discriminator * 100:.2f}%")
print(f"{'Generator Accuracy:':<20} {accuracy_generator * 100:.2f}%")
print(f"{'-' * 30}")



"""
#############################################  
       Structural Similarity Index (SSIM)
 ############################################   
"""

# NO FUNCIONA FALLA AQUÍ: fake_image = fake_image.squeeze(0)  # Eliminar la dimensión del canal (solo si es 1)

def batch_ssim(real_images, fake_images, max_val=1.0):
    """
    Calcula el SSIM promedio entre un lote de imágenes reales y generadas.

    :param real_images: Tensor de imágenes reales (B, C, H, W).
    :param fake_images: Tensor de imágenes generadas (B, C, H, W).
    :param max_val: Valor máximo de los píxeles (por ejemplo, 1.0 si las imágenes están normalizadas entre [0, 1]).
    :return: El SSIM promedio para el lote de imágenes.
    """
    ssim_scores = []
    batch_size = real_images.size(0)
    
    # Iterar sobre el lote de imágenes
    for i in range(batch_size):
        real_image = real_images[i].cpu().detach().numpy()  # Convierte a NumPy
        fake_image = fake_images[i].cpu().detach().numpy()  # Convierte a NumPy
        
       
        # Si las imágenes son de un solo canal, adaptarlas a la forma adecuada (C, H, W) para SSIM
        if real_image.shape[0] == 1:
            print_green("Image of 1 canal")
            real_image = real_image.squeeze(0)  # Eliminar la dimensión del canal (solo si es 1)
            fake_image = fake_image.squeeze(0)  # Eliminar la dimensión del canal (solo si es 1)

        # Si las imágenes tienen 3 canales (por ejemplo, RGB), solo se deben usar como están (sin eliminar canales)
        elif real_image.shape[0] == 3:
            print_green("Image of 3 canal")
            real_image = np.moveaxis(real_image, 0, -1)  # Convertir a (H, W, C)
            fake_image = np.moveaxis(fake_image, 0, -1)  # Convertir a (H, W, C)

        # Calcular el SSIM entre la imagen real y la generada
        score, _ = ssim(real_image, fake_image, full=True, data_range=max_val)
        ssim_scores.append(score)
    
    # Promedio del SSIM en todo el lote
    average_ssim = np.mean(ssim_scores)
    return average_ssim

# Ejemplo de uso con lotes de imágenes reales y generadas (tensores de PyTorch)
num_images= 10
real_images, _ = next(iter(dataloader))  # Obtener un lote de imágenes reales
fake_images = generate_images(netG, num_images, params['nz'])  # Generar un lote de imágenes

# Asegurarte de que las imágenes estén normalizadas en el rango [0, 1]
real_images = (real_images - real_images.min()) / (real_images.max() - real_images.min())
fake_images = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min())

# Calcular SSIM promedio en el lote
ssim_score = batch_ssim(real_images, fake_images)
print(f"SSIM Promedio del lote: {ssim_score:.4f}")
