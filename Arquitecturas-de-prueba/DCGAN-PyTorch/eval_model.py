import torch, json, os
import lpips
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.nn import functional as F
from dcgan import Generator as GeneratorDC, Discriminator as DiscriminatorDC
from dcgan512 import Generator as GeneratorDC512, Discriminator as DiscriminatorDC512
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from utils import get_chestct, get_NBIA
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from tqdm import tqdm

def print_green(text):
    print("\033[92m" + text + "\033[0m")

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

def load_model(model_path, device, model_type, model_name):
    checkpoint = torch.load(f'{model_path}/{model_name}', map_location=device)
    params = checkpoint['params']
    if model_type == 'dcgan':
        if params['imsize'] == 512:
            netG = GeneratorDC512(params).to(device)
            netD = DiscriminatorDC512(params).to(device)
        else:
            netG = GeneratorDC(params).to(device)
            netD = DiscriminatorDC(params).to(device)
    elif model_type == 'wgan':
        """
        """
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    netG.eval()
    netD.eval()
    return netG, netD, params

def evaluate_models(netG, netD, dataloader, device, params):
    correct_discriminator = 0
    total = 0
    generator_confidence = 0  # Suma de las salidas del discriminador sobre imágenes generadas
    
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            
            # Evaluar el discriminador con imágenes reales
            real_labels = torch.ones(b_size, device=device)  # 1 para reales
            output_real = netD(real_data).view(-1)
            correct_discriminator += (output_real.round() == real_labels).sum().item()
            
            # Evaluar el discriminador con imágenes falsas
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            fake_labels = torch.zeros(b_size, device=device)  # 0 para falsas
            output_fake = netD(fake_data.detach()).view(-1)
            correct_discriminator += (output_fake.round() == fake_labels).sum().item()
            
            total += b_size * 2  # Total de ejemplos evaluados

        accuracy_discriminator = correct_discriminator / total  # Precisión del discriminador

        # Evaluación del generador con 1000 ejemplos en lugar de 100
        num_samples = 1000
        noise = torch.randn(num_samples, params['nz'], 1, 1, device=device)
        generated_data = netG(noise)
        output = netD(generated_data).view(-1)  # Predicciones del discriminador
        
        generator_confidence = output.mean().item()  # Promedio de confianza del discriminador en imágenes generadas

    return accuracy_discriminator, generator_confidence

def calculate_ssim(real_images, fake_images):
    real_images = real_images.squeeze().cpu().numpy()
    fake_images = fake_images.squeeze().cpu().numpy()
    fake_images_resized = np.resize(fake_images, real_images.shape)
    ssim_values = [ssim(real, fake, data_range=2.0) for real, fake in zip(real_images, fake_images_resized)]
    return np.mean(ssim_values)

def evaluate_ssim(dataloader, netG, device):
    real_images, fake_images = [], []
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            real_images.append(real_data)
            noise = torch.randn(real_data.size(0), 100, 1, 1, device=device)
            fake_images.append(netG(noise))
    return calculate_ssim(torch.cat(real_images), torch.cat(fake_images))

def calculate_psnr(real, fake):
    mse = F.mse_loss(fake, real)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse > 0 else 100

def evaluate_psnr(dataloader, netG, device, params):
    psnr_total, num_batches = 0, 0
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            noise = torch.randn(real_data.size(0), params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            psnr_total += calculate_psnr(real_data / 2 + 0.5, fake_data / 2 + 0.5)
            num_batches += 1
    return psnr_total / num_batches

def eval_lpips(dataloader, netG, device, imsize):

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])

    # Cargar dataset de imágenes reales
    # real_dataset = datasets.ImageFolder(root=f"{config['datasets']['chestKaggle']}/valid", transform=transform)
    real_dataset = datasets.ImageFolder(root=f"{config['datasets']['nbia']}", transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=True)

    # Obtener una imagen real del dataset
    real_image, _ = next(iter(real_dataloader))  
    real_image = real_image.to(device)

    latent_dim = 100  
    z = torch.randn(1, latent_dim, 1, 1, device=device)  
    generated_image = netG(z) 

    # Calcular LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(real_image, generated_image).item()

    return lpips_value

def main(dataset="nbia", model_name="model_ChestCT.pth"):
    print_green("Evaluating model...")
    model_path = "model_prueba/model_dcgan/"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device, " will be used.\n")
    
    # Cargar modelo
    netG, netD, params = load_model(model_path, device, "dcgan", model_name)
    print_green(str(params))
    
    # Obtener el dataloader usando get_chestct
    if dataset == "chestct":
        dataloader = get_chestct(params["imsize"], bsize=params["bsize"])
    elif dataset == "nbia":
        dataloader = get_NBIA(params["imsize"], bsize=params["bsize"])
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    # Evaluar los modelos
    accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, device, params)
    print(f"Discriminator accuracy: {accuracy_discriminator * 100:.2f}%")
    print(f"Generator confidence: {accuracy_generator * 100:.2f}%")
    
    # Evaluar SSIM, PSNR y LPIPS
    ssim_score = evaluate_ssim(dataloader, netG, device)
    psnr_score = evaluate_psnr(dataloader, netG, device, params)
    lpips_score = eval_lpips(dataloader, netG, device, params["imsize"])
    
    print(f"SSIM: {ssim_score:.4f}")
    print(f"PSNR: {psnr_score:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a GAN model")
    parser.add_argument("--dataset", type=str, default="nbia", choices=["nbia", "chestct"], help="Dataset to use for evaluation")
    parser.add_argument("--model_name", type=str, default="model_ChestCT.pth", help="Name of the model checkpoint to load")
    args = parser.parse_args()

    main(dataset=args.dataset, model_name=args.model_name)
