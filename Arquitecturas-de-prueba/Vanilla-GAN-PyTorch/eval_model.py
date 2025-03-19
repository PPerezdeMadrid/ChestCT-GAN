import torch
import numpy as np
import os
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
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

# Evaluación de métricas
ssim_scores = []
psnr_scores = []
lpips_scores = []

for i in range(10):
    real_img = real_images[i].cpu().numpy().squeeze()
    fake_img = fake_images[i].squeeze()

    # Normalizar imágenes para SSIM y PSNR
    real_img = img_as_float(real_img)
    fake_img = img_as_float(fake_img)

    # SSIM
    ssim_value = ssim(real_img, fake_img, data_range=1.0)
    ssim_scores.append(ssim_value)

    # PSNR
    mse = np.mean((real_img - fake_img) ** 2)
    psnr_value = 10 * np.log10(1.0 / mse) if mse > 0 else 100
    psnr_scores.append(psnr_value)

    # LPIPS (normalizar a [-1,1] antes de pasar al modelo)
    real_img_tensor = torch.tensor(real_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    fake_img_tensor = torch.tensor(fake_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    real_img_tensor = (real_img_tensor - 0.5) * 2
    fake_img_tensor = (fake_img_tensor - 0.5) * 2

    lpips_value = lpips_model(real_img_tensor, fake_img_tensor)
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
