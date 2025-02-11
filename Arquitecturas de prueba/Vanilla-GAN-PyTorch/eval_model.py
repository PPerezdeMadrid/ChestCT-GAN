import torch
import numpy as np
import os
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from skimage.restoration import estimate_sigma
from utils import get_chestct
from gan import Generator, Discriminator
import lpips  # Importar la librería LPIPS

# Cargar configuración
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
n_noise = 100
model_path = "../../../models/model_gan_1000.pth"

# Configurar el dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}")

# Cargar el modelo entrenado
checkpoint = torch.load(model_path, map_location=DEVICE)

G = Generator(n_noise).to(DEVICE)
D = Discriminator().to(DEVICE)
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])

G.eval()
D.eval()

criterion = nn.BCELoss()

# Cargar datos reales para comparación
data_loader = get_chestct()
real_images, _ = next(iter(data_loader))
real_images = real_images[:10].to(DEVICE)  # Seleccionamos 10 imágenes reales para comparar

# Generar imágenes sintéticas
z = torch.randn(10, n_noise).to(DEVICE)
fake_images = G(z).detach().cpu().numpy()

# Evaluar pérdidas
D_real_loss = criterion(D(real_images), torch.ones(real_images.size(0), 1).to(DEVICE)).item()
D_fake_loss = criterion(D(torch.tensor(fake_images).to(DEVICE)), torch.zeros(fake_images.shape[0], 1).to(DEVICE)).item()
D_loss = D_real_loss + D_fake_loss

G_loss = criterion(D(torch.tensor(fake_images).to(DEVICE)), torch.ones(fake_images.shape[0], 1).to(DEVICE)).item()

# Inicializar el modelo LPIPS
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)  # Puedes usar 'vgg' o 'alex' como red base

# Evaluación de métricas
ssim_scores = []
psnr_scores = []
lisps_scores = []

for i in range(10):
    real_img = real_images[i].cpu().numpy().squeeze()
    fake_img = fake_images[i].squeeze()

    # Normalizar imágenes para SSIM y PSNR
    real_img = img_as_float(real_img)
    fake_img = img_as_float(fake_img)

    # SSIM
    ssim_value = ssim(real_img, fake_img, data_range=fake_img.max() - fake_img.min())
    ssim_scores.append(ssim_value)

    # PSNR
    mse = np.mean((real_img - fake_img) ** 2)
    psnr_value = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
    psnr_scores.append(psnr_value)

    # LPIPS
    real_img_tensor = torch.tensor(real_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    fake_img_tensor = torch.tensor(fake_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # Calcular LPIPS
    lpips_value = lpips_model(real_img_tensor, fake_img_tensor)
    lisps_scores.append(lpips_value.item())

# Imprimir resultados
# Imprimir resultados con el formato solicitado
print("------------------------------")
print("Model Evaluation Results")
print("------------------------------")
print(f"Discriminator Accuracy: {100 * (1 - D_loss):.2f}%")
print(f"Generator Accuracy: {100 * (1 - G_loss):.2f}%")
print("------------------------------")
print(f"SSIM Score: {np.mean(ssim_scores):.4f}")
print("------------------------------")
print(f"PSNR Score: {np.mean(psnr_scores):.2f} dB")
print("------------------------------")
print(f"LPIPS Score: {np.mean(lisps_scores):.4f}")
print("------------------------------")


# Mostrar algunas imágenes reales y generadas
fig, axs = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axs[0, i].imshow(real_images[i].cpu().numpy().squeeze(), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("Real")

    axs[1, i].imshow(fake_images[i].squeeze(), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title("Generated")

plt.show()
