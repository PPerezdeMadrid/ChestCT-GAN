import torch
import numpy as np
import os
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from skimage.restoration import estimate_sigma
from utils import get_xray
from dcgan import Generator, Discriminator
import lpips  # Importar la librería LPIPS

# Cargar configuración
with open('config.json', 'r') as json_file:
    config = json.load(json_file)


model_path = f"{config['model']['path_dcgan']}"

# Configurar el dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}")

# Cargar el modelo entrenado con weights_only=True para una mayor seguridad
# model = "model_ChestCT.pth"
model = "model_epoch_400.pth"
checkpoint = torch.load(f'{model_path}/{model}',  map_location=DEVICE)
params = checkpoint['params']

# Get the data
dataloader = get_xray(params['imsize'], params['bsize'])

# Load the generator and discriminator
netG = Generator(params).to(DEVICE)
netD = Discriminator(params).to(DEVICE)

netG.load_state_dict(checkpoint['generator'])
netD.load_state_dict(checkpoint['discriminator'])


# Modo evaluación
netG.eval()
netD.eval()

""""
##############################
    Discriminador y Generador
######################
"""


def evaluate_models(netG, netD, dataloader, device, params, num_samples=1000):
    
    correct_discriminator = 0
    total = 0
    correct_generator = 0

    with torch.no_grad():
        """
        Asigna una probabilidad mayor a 0.5 a imágenes reales.
        Asigna una probabilidad menor a 0.5 a imágenes falsas.
        """
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)

            # Evaluar el discriminador en datos reales
            real_label_tensor = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_data).view(-1)
            # for prob in output_real:
                # print(f"Dato real clasificado como --> {prob.item()}")

            correct_discriminator += ((output_real > 0.5).float() == real_label_tensor).sum().item()

            # Generar datos falsos
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)

            # Evaluar el discriminador en datos generados
            fake_label_tensor = torch.full((b_size,), 0, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach()).view(-1)
            #for prob in output_fake:
                # print(f"Dato falso clasificado como --> {prob.item()}")

            correct_discriminator += ((output_fake < 0.5).float() == fake_label_tensor).sum().item()

            total += b_size * 2  # Para datos reales y generados

        # Calcular precisión del discriminador
        accuracy_discriminator = correct_discriminator / total


        """
        Imágenes generadas por el generador y etiquetadas reales
        """
        for _ in range(num_samples): 
            noise = torch.randn(1, params['nz'], 1, 1, device=device)
            generated_data = netG(noise)
            output = netD(generated_data.detach()).view(-1)
            print(f"Probabilidad de que la imagen generada sea real: {output.item()}")
            if output > 0.5:
                correct_generator += 1

        accuracy_generator = correct_generator / 1000

    return accuracy_discriminator, accuracy_generator


accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, DEVICE, params)


print(f"{'-' * 30}")
print(f"{'Model Evaluation Results':^30}")
print(f"{'-' * 30}")
print(f"{'Discriminator Accuracy:':<20} {accuracy_discriminator * 100:.2f}%")
print(f"{'  Generator Accuracy:':<20} {accuracy_generator * 100:.2f}%")
print(f"{'-' * 30}")

def visualize_samples(netG, device, params, num_samples=5):
    netG.eval()
    noise = torch.randn(num_samples, params['nz'], 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(noise).cpu().numpy()

    fig, axs = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axs[i].imshow(fake_images[i, 0], cmap='gray')
        axs[i].axis('off')
    plt.show()


def evaluate_generator_prob(netG, netD, device, params, num_samples=100):
    netG.eval()
    netD.eval()
    
    total_real_prob = 0  

    with torch.no_grad():
        noise = torch.randn(num_samples, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)  # Generar imágenes
        output = torch.sigmoid(netD(fake_data)).view(-1)  # Obtener probabilidad de ser real, sigmoid porque queremos probabilidades interpretables. (hemos usado BCLoss)
        
        total_real_prob = output.mean().item()  # Promediar probabilidad

    return total_real_prob  

def evaluate_discriminator_prob(netG, netD, device, params, num_samples=100):
    netG.eval()
    netD.eval()

    correct_discriminator = 0
    total = 0

    with torch.no_grad():
        # Evaluar discriminador sobre imágenes reales
        real_data, _ = next(iter(dataloader))  # Obtenemos un batch de imágenes reales
        real_data = real_data.to(device)
        real_label = torch.ones(real_data.size(0), 1, device=device)

        output_real = torch.sigmoid(netD(real_data)).view(-1)
        correct_discriminator += (output_real > 0.5).sum().item()

        total += real_data.size(0)

        # Evaluar discriminador sobre imágenes generadas
        noise = torch.randn(num_samples, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)
        fake_label = torch.zeros(fake_data.size(0), 1, device=device)

        output_fake = torch.sigmoid(netD(fake_data)).view(-1)
        correct_discriminator += (output_fake < 0.5).sum().item()

        total += fake_data.size(0)

    # Calcular la precisión del discriminador
    discriminator_accuracy = correct_discriminator / total
    return discriminator_accuracy



visualize_samples(netG, DEVICE, params)

num_samples = 1000
generator_real_prob = evaluate_generator_prob(netG, netD, DEVICE, params, num_samples=num_samples)
discriminator_accuracy = evaluate_discriminator_prob(netG, netD, DEVICE, params, num_samples=num_samples)

# Imprimir resultados
print(f"{'-' * 30}")
print(f"With {num_samples} samples, the discriminator and generator have been evaluated:")
print(f"{'Discriminator Accuracy Probability :':<20} {discriminator_accuracy * 100:.2f}%")
print(f"{'Generator Accuracy Probability:':<20} {generator_real_prob * 100:.2f}%")
print(f"{'-' * 30}")

