import torch, json, lpips, argparse
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
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt



def print_green(text):
    print("\033[92m" + text + "\033[0m")

print_green("Evaluating model...")
parser = argparse.ArgumentParser(description='Model selection for ChestCT')
parser.add_argument('--model', type=str, choices=['dcgan', 'wgan'], default="dcgan", help='Select model type: dcgan or wgan')
args = parser.parse_args()

with open('config.json', 'r') as json_file:
    config = json.load(json_file)


if args.model == 'dcgan':
    print_green("Evaluating DCGAN...")
    model_path = config["model"]["path_dcgan"]
    image_path = config["model"]["image_path_dcgan"]
elif args.model == 'wgan':
    print_green("Evaluating WGAN...")
    model_path = config["model"]["path_wgan"]
    image_path = config["model"]["image_path_wgan"]

print_green("Parameters uploaded")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print(device, " will be used.\n")


# Cargar el modelo entrenado con weights_only=True para una mayor seguridad
checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', weights_only=True)
params = checkpoint['params']

# Get the data
dataloader = get_chestct(params['imsize'])

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

def calculate_ssim(real_images, fake_images):
    """Calcula el índice SSIM entre imágenes reales y generadas."""
    real_images = real_images.squeeze().cpu().numpy()  # Eliminar el canal adicional (si es necesario)
    fake_images = fake_images.squeeze().cpu().numpy()  # Eliminar el canal adicional (si es necesario)

    # Asegurarse de que las imágenes tengan el mismo tamaño
    fake_images_resized = np.resize(fake_images, real_images.shape)

    # Calcular SSIM para cada par de imágenes
    ssim_values = []
    for real, fake in zip(real_images, fake_images_resized):
        # Aquí definimos el data_range como 2, ya que las imágenes están en [-1, 1]
        ssim_value = ssim(real, fake, data_range=2.0)
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)  # Devolver el valor promedio del SSIM

def evaluate_ssim(dataloader, netG, device):
    """Evaluar SSIM entre imágenes reales y generadas."""
    real_images = []
    fake_images = []
    
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)

            # Obtener imágenes reales
            real_images.append(real_data)

            # Generar imágenes falsas
            noise = torch.randn(real_data.size(0), 100, 1, 1, device=device)  # Ajustar según tu tamaño de latente
            fake_data = netG(noise)
            fake_images.append(fake_data)
        
    # Convertir listas a tensores
    real_images = torch.cat(real_images, dim=0)
    fake_images = torch.cat(fake_images, dim=0)
    
    # Calcular SSIM
    ssim_score = calculate_ssim(real_images, fake_images)
    return ssim_score

# Evaluar el SSIM
ssim_score = evaluate_ssim(dataloader, netG, device)
# Imprimir --> SSIM 
print(f"{'-' * 30}")
print(f"{'SSIM Score:':<20} {ssim_score:.4f}")

"""
#############################################  
    Peak Signal-to-Noise Ratio (PSNR)
############################################   
"""

def calculate_psnr(real, fake, max_pixel=1.0):
    """
    Calcula el PSNR (Peak Signal-to-Noise Ratio) entre las imágenes reales y generadas.
    
    Args:
    - real (tensor): Imagen real (en tensor).
    - fake (tensor): Imagen generada (en tensor).
    - max_pixel (float): El valor máximo posible de los píxeles (por ejemplo, 1.0 para imágenes normalizadas).
    
    Returns:
    - psnr (float): El valor del PSNR.
    """
    # Calcular MSE (Mean Squared Error)
    mse = F.mse_loss(fake, real)
    
    # Calcular PSNR a partir del MSE
    if mse == 0:
        return 100  # Si el MSE es 0, PSNR es infinito (imágenes idénticas)
    
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# Función de evaluación con PSNR
def evaluate_psnr(dataloader, netG, device, params):
    psnr_total = 0
    num_batches = 0
    
    # Evaluar el modelo en lotes
    with torch.no_grad():
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            
            # Generar datos falsos
            noise = torch.randn(real_data.size(0), params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            
            # Asegurarse de que las imágenes tengan el mismo rango de valores
            # Normalizar las imágenes entre [0, 1] si es necesario (dependiendo de cómo estén las imágenes)
            fake_data = fake_data / 2 + 0.5  # Para asegurarse de que las imágenes estén entre [0, 1]
            real_data = real_data / 2 + 0.5  # Para asegurarse de que las imágenes estén entre [0, 1]
            
            # Calcular PSNR para este batch
            psnr_batch = calculate_psnr(real_data, fake_data)
            psnr_total += psnr_batch
            num_batches += 1
    
    # Promedio de PSNR sobre todos los lotes
    average_psnr = psnr_total / num_batches
    return average_psnr

# Ejemplo de cómo usarlo
psnr_score = evaluate_psnr(dataloader, netG, device, params)
# Imprimir --> SSIM 
print(f"{'-' * 30}")
print(f"{'PSNR Score:':<20} {psnr_score:.2f} dB")
print(f"{'-' * 30}")


"""
#############################################  
                  LPIPS 
############################################   
"""

if not torch.cuda.is_available():
    print("CUDA no disponible. Se calculará el valor LISP con la CPU")
    # Cargar el modelo LPIPS
    loss_fn = lpips.LPIPS(net='alex')  # Funciona sin cuda

    # Si estás usando GPU, mover el modelo a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = loss_fn.to(device)

    def calculate_lpips(real_image, generated_image):
        # Asegúrate de que las imágenes tengan el formato correcto
        real_image = real_image.unsqueeze(0).float().to(device)  # Añadir batch y mover a GPU/CPU
        generated_image = generated_image.unsqueeze(0).float().to(device)

        # Calcular la distancia LPIPS
        distance = loss_fn(real_image, generated_image)
        return distance.item()

    def load_image(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Redimensionar a 224x224 (tamaño esperado por VGG/ResNet)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    # Cargar las imágenes
    real_image = load_image('../../../ChestCTKaggle/Data/valid/normal/5.png')
    generated_image = load_image(f'{image_path}/generated_image_30.png')

    # Calcular el valor LPIPS
    lpips_value = calculate_lpips(real_image, generated_image)
    print(f"LPIPS: {lpips_value:.4f}")


else:
    print("CUDA disponible. El modelo se ejecutará en la GPU.")

    lpips_model = lpips.LPIPS(net='vgg').cuda() # Más optimo

    # Función para cargar y preprocesar imágenes
    def load_image(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).cuda()

    # Cargar imágenes de ejemplo
    image1 = load_image(f'{image_path}/generated_image_30.png')
    image2 = load_image('../../../ChestCTKaggle/Data/valid/normal/5.png')


    lpips_score = lpips_model(image1, image2)
    print(f"LPIPS Score: {lpips_score.item():.4f}")








