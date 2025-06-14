import torch, json, os
import lpips
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dcgan import Generator as GeneratorDC, Discriminator as DiscriminatorDC
from dcgan512 import Generator as GeneratorDC512, Discriminator as DiscriminatorDC512
from dcgan256 import Generator as GeneratorDC256, Discriminator as DiscriminatorDC256
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from utils import get_chestct, get_NBIA
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from scipy.stats import entropy


def print_green(text):
    print("\033[92m" + text + "\033[0m")

def load_config(config_file='config.json'):
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    return config


def load_model(model_path, device, model_type, model_name):
    checkpoint = torch.load(f'{model_path}/{model_name}', map_location=device)
    params = checkpoint['params']
    if model_type == 'dcgan':
        if params['imsize'] == 512:
            netG = GeneratorDC512(params).to(device)
            netD = DiscriminatorDC512(params).to(device)
        elif params['imsize'] == 256:
            netG = GeneratorDC256(params).to(device)
            netD = DiscriminatorDC256(params).to(device)
        elif params['imsize'] == 64:
            netG = GeneratorDC(params).to(device)
            netD = DiscriminatorDC(params).to(device)
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
    generator_confidence = 0  
    
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            b_size = real_data.size(0)
            
            # evaluate the discriminator with real images
            real_labels = torch.ones(b_size, device=device)  # 1 for real
            output_real = netD(real_data).view(-1)
            correct_discriminator += (output_real.round() == real_labels).sum().item()
            
            # Evaluate the discriminator with fake images
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            fake_labels = torch.zeros(b_size, device=device)  # 0 for fake
            output_fake = netD(fake_data.detach()).view(-1)
            correct_discriminator += (output_fake.round() == fake_labels).sum().item()
            
            total += b_size * 2  # total images (real + fake)

        accuracy_discriminator = correct_discriminator / total  # Discriminator accuracy

        # Evaluate generator confidence
        num_samples = 1000
        noise = torch.randn(num_samples, params['nz'], 1, 1, device=device)
        generated_data = netG(noise)
        output = netD(generated_data).view(-1)  # Discriminator output for generated images
        
        generator_confidence = output.mean().item()  # Mean confidence of the generator

    return accuracy_discriminator, generator_confidence

def calculate_ssim(real_images, fake_images):
    real_images = real_images.squeeze().cpu().numpy()
    fake_images = fake_images.squeeze().cpu().numpy()
    fake_images_resized = np.resize(fake_images, real_images.shape)
    ssim_values = [ssim(real, fake, data_range=2.0) for real, fake in zip(real_images, fake_images_resized)]
    return np.mean(ssim_values)

def evaluate_ssim(dataloader, netG, device, params):
    real_images, fake_images = [], []
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            real_images.append(real_data)
            noise = torch.randn(real_data.size(0), params['nz'], 1, 1, device=device)
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

def eval_lpips(dataloader, netG, device, params, configFile):
    config = load_config(configFile)

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    transform = transforms.Compose([
        transforms.Resize((params["imsize"], params["imsize"])),
        transforms.ToTensor()
    ])

    # load real images from the dataset
    # real_dataset = datasets.ImageFolder(root=f"{config['datasets']['chestKaggle']}/valid", transform=transform)
    real_dataset = datasets.ImageFolder(root=f"{config['datasets']['nbia']}", transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=True)

    # get a real image from the dataset
    real_image, _ = next(iter(real_dataloader))  
    real_image = real_image.to(device)
 
    z = torch.randn(1, params["nz"], 1, 1, device=device)  
    generated_image = netG(z) 

    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(real_image, generated_image).item()

    return lpips_value


def calculate_fid(real_images, generated_images):
     # load pre-trained Inception v3 model
     inception_model = models.inception_v3(pretrained=True, transform_input=False)
     inception_model.eval()
     
     transform = transforms.Compose([
         transforms.Resize((299, 299)), # Inception v3 solo trabaja con img de 299x299 y RGB (3 canales)
         transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Típico en Inception v3
     ])
     
     # If the input is a directory, load all images from the directory
     if os.path.isdir(real_images):
        print(f"Loading real images from {real_images}")
        real_images = [os.path.join(real_images, img) for img in os.listdir(real_images) if img.endswith(('jpg', 'png', 'jpeg'))]
     
     if os.path.isdir(generated_images):
        print(f"Loading generated images from {generated_images}")
        generated_images = [os.path.join(generated_images, img) for img in os.listdir(generated_images) if img.endswith(('jpg', 'png', 'jpeg'))]
     
     real_images = [transform(Image.open(img)) if isinstance(img, str) else transform(img) for img in real_images]
     print(f"Real images: {len(real_images)}")
     generated_images = [transform(Image.open(img)) if isinstance(img, str) else transform(img) for img in generated_images]
     print(f"Generated images: {len(generated_images)}")
     
     real_images = torch.stack(real_images)
     generated_images = torch.stack(generated_images)
     
     with torch.no_grad():
         real_features = inception_model(real_images).detach().numpy()
         generated_features = inception_model(generated_images).detach().numpy()
     
     # Calculate mean and covariance of the features
     mu_real = np.mean(real_features, axis=0)
     sigma_real = np.cov(real_features, rowvar=False)
     mu_generated = np.mean(generated_features, axis=0)
     sigma_generated = np.cov(generated_features, rowvar=False)
     
     # Calculate FID
     diff = mu_real - mu_generated
     covmean = sqrtm(sigma_real.dot(sigma_generated))
     
     if np.iscomplexobj(covmean):
         covmean = covmean.real
     
     fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
     return fid
    

def calculate_inception_score(generated_images, device, imsize=299):
    # Load pre-trained Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels for Inception
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    generated_images = [transform(img) if isinstance(img, Image.Image) else transform(Image.fromarray(img)) for img in generated_images]
    generated_images = torch.stack(generated_images).to(device)

    with torch.no_grad():
        preds = F.softmax(inception_model(generated_images), dim=1).cpu().numpy()

    # Calculate Inception Score
    marginal_probs = np.mean(preds, axis=0)
    kl_divergences = [entropy(pred, marginal_probs) for pred in preds]
    inception_score = np.exp(np.mean(kl_divergences))

    return inception_score

def eval_inception_score(netG, device, num_samples=1000, imsize=299, params=None):
    generated_images = []
    latent_dim = params["nz"]  
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            generated_image = netG(z).squeeze(0).cpu()

            # Convert the tensor to a PIL image
            generated_image = transforms.ToPILImage()(generated_image)
            generated_images.append(generated_image)

    return calculate_inception_score(generated_images, device, imsize)

def main(dataset="nbia", model_name="model_ChestCT.pth", discarded=False, configFile="config.json"):
    config = load_config(configFile)
    model_path = config["model"]["path_dcgan"]
    print_green(f'Evaluating model {model_path}/{model_name}...')
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device, " will be used.\n")
    
    # Load the model
    netG, netD, params = load_model(model_path, device, "dcgan", model_name)
    print_green(str(params))
    
    # Get the dataloader
    if dataset == "chestct":
        dataloader = get_chestct(params["imsize"], bsize=params["bsize"])
    elif dataset == "nbia":
        dataloader = get_NBIA(params["imsize"], bsize=params["bsize"])
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    real_images = f"{config["datasets"][dataset]}/cancer"
    generated_images = f"{config["model"]["image_path_dcgan"]}/generated_{params["imsize"]}"
    
    
    if discarded:
        fid_score = calculate_fid(real_images, generated_images)
        inception_score = eval_inception_score(netG, device, params=params)


    # Evaluate the models
    accuracy_discriminator, accuracy_generator = evaluate_models(netG, netD, dataloader, device, params)
    
    # Evaluate SSIM, PSNR and LPIPS
    ssim_score = evaluate_ssim(dataloader, netG, device, params)
    psnr_score = evaluate_psnr(dataloader, netG, device, params)
    lpips_score = eval_lpips(dataloader, netG, device, params, configFile)
    
    print(f"{'-' * 30}")
    print(f"{'Model Evaluation Results':^30}")
    print(f"{'-' * 30}")
    print(f"{'Discriminator Accuracy:':<20} {accuracy_discriminator * 100:.2f}%")
    print(f"{'Generator Accuracy:':<20} {accuracy_generator * 100:.2f}%")
    print(f"{'SSIM Score:':<20} {ssim_score:.4f}")
    print(f"{'PSNR Score:':<20} {psnr_score:.4f}")
    print(f"{'LPIPS Score':<20} {lpips_score:.4f}")
    if discarded:
        print(f"{'FID Score':<20} {fid_score:.4f}")
        print(f"{'Inception Score ':<20} {inception_score:.4f}")
    print(f"{'-' * 30}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a GAN model")
    parser.add_argument("--dataset", type=str, default="nbia", choices=["nbia", "chestct"], help="Dataset to use for evaluation")
    parser.add_argument("--model_name", type=str, default="model_ChestCT.pth", help="Name of the model checkpoint to load")
    parser.add_argument("--discarded", action="store_true", help="Show discarded metrics info (IS, FID, Precision & Recall for GANs)")
    parser.add_argument("--configFile", type=str, default="config.json", help="Path to the config file")
    args = parser.parse_args()

    main(dataset=args.dataset, model_name=args.model_name, discarded=args.discarded, configFile=args.configFile)
