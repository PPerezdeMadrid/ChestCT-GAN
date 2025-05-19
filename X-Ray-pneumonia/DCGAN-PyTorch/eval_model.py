import torch
import numpy as np
import os
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from utils import get_xray
from dcgan import Generator, Discriminator
import lpips  

# Cargar configuración
with open('config.json', 'r') as json_file:
    config = json.load(json_file)


model_name = "model_epoch_400.pth"
model_path = f"{config["model"]["path_dcgan"]}/{model_name}"


# Configurar el dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}")

# Cargar el modelo entrenado
checkpoint = torch.load(model_path, map_location=DEVICE)
params = checkpoint['params']
n_noise = params["nz"]

G = Generator(params).to(DEVICE)
D = Discriminator(params).to(DEVICE)
G.load_state_dict(checkpoint['generator'])
D.load_state_dict(checkpoint['discriminator'])

G.eval()
D.eval()

criterion = nn.BCELoss()

# Cargar datos reales para comparación
data_loader = get_xray(img_size=params["imsize"], bsize=params["bsize"])
real_images, _ = next(iter(data_loader))
real_images = real_images[:10].to(DEVICE)  # Seleccionamos 10 imágenes reales para comparar

# Generar imágenes sintéticas
z = torch.randn(10, params["nz"]).to(DEVICE)
fake_images = G(z).detach()

# Normalizar imágenes generadas si el generador usa tanh
fake_images = (fake_images + 1) / 2  # Convertir de [-1,1] a [0,1]
fake_images = fake_images.cpu().numpy()

# Evaluar pérdidas
D_real_loss = criterion(D(real_images), torch.ones(real_images.size(0), 1).to(DEVICE)).item()
D_fake_loss = criterion(D(torch.tensor(fake_images).to(DEVICE)), torch.zeros(fake_images.shape[0], 1).to(DEVICE)).item()
D_loss = D_real_loss + D_fake_loss

G_loss = criterion(D(torch.tensor(fake_images).to(DEVICE)), torch.ones(fake_images.shape[0], 1).to(DEVICE)).item()

# Calcular precisión del discriminador y generador
with torch.no_grad():
    real_preds = D(real_images) > 0.5
    fake_preds = D(torch.tensor(fake_images).to(DEVICE)) < 0.5

discriminator_accuracy = 100 * torch.cat((real_preds, fake_preds), dim=0).float().mean().item()
generator_accuracy = 100 * (D(torch.tensor(fake_images).to(DEVICE)) > 0.5).float().mean().item()

# Inicializar el modelo LPIPS
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

# Inicializar listas para las métricas
ssim_scores = []
psnr_scores = []
lpips_scores = []

# Evaluación de métricas
for i in range(real_images.size(0)):  # Para cada imagen
    # Calcular SSIM
    real_image = real_images[i].cpu().numpy().squeeze()
    fake_image = fake_images[i].squeeze()
    
    ssim_score = ssim(real_image, fake_image, data_range=1)
    ssim_scores.append(ssim_score)
    
    # Calcular PSNR
    psnr_score = 10 * np.log10(1 / np.mean((real_image - fake_image) ** 2))
    psnr_scores.append(psnr_score)
    
    # Calcular LPIPS
    lpips_value = lpips_model(real_images[i:i+1], torch.tensor(fake_images[i:i+1]).to(DEVICE))  # LPIPS acepta batch
    lpips_scores.append(lpips_value.item())


# Imprimir resultados con el formato corregido
print("------------------------------")
print("Model Evaluation Results")
print("------------------------------")
print(f"Discriminator Accuracy: {discriminator_accuracy:.2f}%")
print(f"Generator Accuracy: {generator_accuracy:.2f}%")
print("------------------------------")
print(f"SSIM Score: {np.mean(ssim_scores):.4f}")
print("------------------------------")
print(f"PSNR Score: {np.mean(psnr_scores):.2f} dB")
print("------------------------------")
print(f"LPIPS Score: {np.mean(lpips_scores):.4f}")
print("------------------------------")

fig, axs = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axs[0, i].imshow(real_images[i].cpu().numpy().squeeze(), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("Real")

    axs[1, i].imshow(fake_images[i].squeeze(), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title("Generated")

plt.show()
