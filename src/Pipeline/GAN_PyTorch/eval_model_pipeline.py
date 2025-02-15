import torch,json,lpips
import torchvision.transforms as transforms
from torchvision import models
from torch.nn import functional as F
from GAN_PyTorch.dcgan import Generator as GeneratorDC, Discriminator as DiscriminatorDC
from GAN_PyTorch.wgan import Generator as GeneratorW, Discriminator as DiscriminatorW
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import get_chestct
from datetime import datetime

def print_green(text):
    print("\033[92m" + text + "\033[0m")

def load_config(config_path):
    with open(config_path, 'r') as json_file:
        return json.load(json_file)

def setup_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def load_model(model_path, device, model_type):
    checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', map_location=device)
    params = checkpoint['params']
    if model_type == 'dcgan':
        netG = GeneratorDC(params).to(device)
        netD = DiscriminatorDC(params).to(device)
    elif model_type == 'wgan':
        netG = GeneratorW(params).to(device)
        netD = DiscriminatorW(params).to(device)
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    netG.eval()
    netD.eval()
    return netG, netD, params

def evaluate_models(netG, netD, dataloader, device, params):
    correct_discriminator = 0
    total = 0
    correct_generator = 0

    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            real_label_tensor = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_data).view(-1)
            correct_discriminator += ((output_real > 0.5).float() == real_label_tensor).sum().item()

            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            fake_label_tensor = torch.full((b_size,), 0, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach()).view(-1)
            correct_discriminator += ((output_fake < 0.5).float() == fake_label_tensor).sum().item()
            total += b_size * 2

        accuracy_discriminator = correct_discriminator / total
        for _ in range(100):
            noise = torch.randn(1, params['nz'], 1, 1, device=device)
            generated_data = netG(noise)
            output = netD(generated_data.detach()).view(-1)
            if output > 0.5:
                correct_generator += 1

        accuracy_generator = correct_generator / 100
    
    return accuracy_discriminator, accuracy_generator

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

def calculate_lpips(model, img1_path, img2_path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img1, img2 = Image.open(img1_path).convert('RGB'), Image.open(img2_path).convert('RGB')
    img1, img2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
    return model(img1, img2).item()

def validate_eval(accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score):
    """Calcula una puntuación del 1 y 10 basado en las métricas del modelo"""

    score_discriminator = (accuracy_discriminator * 100) / 10
    score_generator = (accuracy_generator * 100) / 10

    score_ssim = ssim_score * 10
    
    if psnr_score >= 30:
        score_psnr = 10
    elif psnr_score >= 25:
        score_psnr = 8
    elif psnr_score >= 20:
        score_psnr = 6
    else:
        score_psnr = 4
 
    score_lpips = max(0, 10 - (lpips_score * 20))  # Invertir LPIPS: cuanto menor, mejor

    # Calificación final: Promedio de todas las puntuaciones
    puntuacion_total = (score_discriminator + score_generator + score_ssim + score_psnr + score_lpips) / 5
    puntuacion_final = round(puntuacion_total)
    
    return puntuacion_final




def main(model_type):
    print_green("Evaluating model...")
    config = load_config("GAN_PyTorch/config.json")
    model_path = config["model"][f"path_{model_type}"]
    report_path = config["model"][f'evaluation_{model_type}']
    device = setup_device()
    print(device, " will be used.\n")
    netG, netD, params = load_model(model_path, device, model_type)
    dataloader = get_chestct(params['imsize'])
    
    accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, device, params)
    ssim_score = evaluate_ssim(dataloader, netG, device)
    psnr_score = evaluate_psnr(dataloader, netG, device, params)
    model_lpips = lpips.LPIPS(net='vgg').to(device)
    img_generated = f'../Data/images/images_{model_type}/img_eval_lpips.png' # Se necesita generar 1 imagen al menos para evaluar
    lpips_score = calculate_lpips(model_lpips, '../Data/Imagen_Ref1.png', img_generated)
    
    date = datetime.now().strftime('%Y-%m-%d')
    report_name = f'EvalModel_{model_type}_{date}.md'
    output_path = f'{report_path}/{report_name}'

    with open('template_EvalModel.md', 'r', encoding='utf-8') as file:
        template = file.read()

    # Reemplazar las variables en la plantilla
    report_content = template.format(
        model_type=model_type,
        accuracy_discriminator=accuracy_discriminator,
        accuracy_generator=accuracy_generator,
        ssim_score=ssim_score,
        psnr_score=psnr_score,
        lpips_score=lpips_score,
    )
    with open(output_path, "w") as md_file:
        md_file.write(report_content)
    print(f"Report saved to {output_path}")
    return accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score, output_path

