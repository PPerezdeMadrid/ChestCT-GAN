import torch, json, os
import lpips
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from torch.nn import functional as F
from wgan import Generator , Discriminator
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from utils import get_chestct
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from scipy.stats import entropy


def print_green(text):
    print("\033[92m" + text + "\033[0m")

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

def load_model(model_path, device, model_type, model_name):
    checkpoint = torch.load(f'{model_path}/{model_name}', map_location=device)
    params = checkpoint['params']
    if model_type == 'dcgan':
        """
        """
    elif model_type == 'wgan':
        netG = Generator(params).to(device)
        netD = Discriminator(params).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    netG.eval()
    netD.eval()
    return netG, netD, params


def evaluate_models(netG, netD, dataloader, device, params):
    correct_discriminator = 0
    total = 0
    correct_generator = 0
    discriminator_threshold = 0.5  # Umbral for the discriminator

    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)

            # Evaluate the discriminator on real data
            real_label_tensor = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_data).view(-1)
            correct_discriminator += ((output_real > discriminator_threshold).float() == real_label_tensor).sum().item()

            # Generate fake data
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)

            # Evaluate the discriminator on fake data
            fake_label_tensor = torch.full((b_size,), 0, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach()).view(-1)
            correct_discriminator += ((output_fake < discriminator_threshold).float() == fake_label_tensor).sum().item()

            total += b_size * 2  

        # Calculate the accuracy of the discriminator
        accuracy_discriminator = correct_discriminator / total

        # Evaluate the generator
        num_test_samples = 500 
        for _ in range(num_test_samples):  
            noise = torch.randn(1, params['nz'], 1, 1, device=device)
            generated_data = netG(noise)
            output = netD(generated_data.detach()).view(-1)
            if output > discriminator_threshold:
                correct_generator += 1

        accuracy_generator = correct_generator / num_test_samples

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
            noise = torch.randn(real_data.size(0), 128, 1, 1, device=device)
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

def eval_lpips(netG, device, params):

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    transform = transforms.Compose([
        transforms.Resize((params["imsize"], params["imsize"])),
        transforms.ToTensor()
    ])

    # Load the real dataset
    real_dataset = datasets.ImageFolder(root=f"{config['datasets']['chestKaggle']}valid/", transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=True)

    # Get a batch of real images
    real_image, _ = next(iter(real_dataloader))  
    real_image = real_image.to(device)

    latent_dim = params["nz"]  
    z = torch.randn(1, latent_dim, 1, 1, device=device)  
    generated_image = netG(z) 

    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(real_image, generated_image).item()

    return lpips_value



def main(dataset="chestct", model_name="model_ChestCT.pth", config_path = "config.json"):
    print_green("Evaluating model...")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    model_path = config["model"]["path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device, " will be used.\n")
    
    # Load the model
    netG, netD, params = load_model(model_path, device, "wgan", model_name)
    print_green(str(params))
    
    # Get the dataloader
    if dataset == "chestct":
        dataloader = get_chestct()
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")



    # Evaluate the model
    accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, device, params)
    
    # Evaluate SSIM, PSNR and LPIPS
    ssim_score = evaluate_ssim(dataloader, netG, device)
    psnr_score = evaluate_psnr(dataloader, netG, device, params)
    lpips_score = eval_lpips(netG, device, params=params)
    
    print(f"{'-' * 30}")
    print(f"{'Model Evaluation Results':^30}")
    print(f"{'-' * 30}")
    print(f"{'Discriminator Accuracy:':<20} {accuracy_discriminator * 100:.2f}%")
    print(f"{'Generator Accuracy:':<20} {accuracy_generator * 100:.2f}%")
    print(f"{'SSIM Score:':<20} {ssim_score:.4f}")
    print(f"{'PSNR Score:':<20} {psnr_score:.4f}")
    print(f"{'LPIPS Score':<20} {lpips_score:.4f}")
    print(f"{'-' * 30}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a GAN model")
    parser.add_argument("--dataset", type=str, default="chestct", choices=["nbia", "chestct"], help="Dataset to use for evaluation")
    parser.add_argument("--model_name", type=str, default="model_ChestCT.pth", help="Name of the model checkpoint to load")
    args = parser.parse_args()


    main(dataset=args.dataset, model_name=args.model_name)
