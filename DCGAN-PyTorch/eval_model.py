import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.nn import functional as F
from dcgan import Generator  # Asegúrate de que este módulo está definido
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from utils import get_chestct 
from dcgan import Generator, Discriminator
import json


def print_green(text):
    print("\033[92m" + text + "\033[0m")

print_green("Evaluación iniciada.")

# leer parámetros y modelo:
with open('config.json', 'r') as json_file:
    config = json.load(json_file)


model_path = config["model"]["path"]

print_green("Parámetros cargados.")

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
        Inception Score (IS)
 ############################################   
"""

def get_inception_score(images, splits=10):
    """Calcula el Inception Score para un conjunto de imágenes."""
    scores = []
    N = len(images)
    
    for i in range(splits):
        part = images[i * (N // splits): (i + 1) * (N // splits)]
        with torch.no_grad():
            pred = inception_model(part)  # Pasar las imágenes por Inception
            pred = F.softmax(pred, dim=1)  # Aplicar softmax

        scores.append(pred.cpu().numpy())
    
    scores = np.concatenate(scores, axis=0)
    # Calcular el Inception Score
    kl_divergence = scores * (np.log(scores) - np.log(np.mean(scores, axis=0)))
    is_score = np.exp(np.mean(np.sum(kl_divergence, axis=1)))
    
    return is_score

# Generar imágenes
num_images = 100  # Número de imágenes a generar
fake_images = generate_images(netG, num_images, params['nz'])

# Preprocesar imágenes
preprocessed_images = preprocess_images(fake_images)

# Cargar Inception-v3
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

# Calcular el Inception Score
is_score = get_inception_score(preprocessed_images)
# Print pero en bonito :)
print(f"{'-' * 30}")
print(f"{'Inception Score':^30}")
print(f"{'-' * 30}")
print(f"{'Score:':<20} {is_score:.4f}") 
print(f"{'-' * 30}")

"""
#############################################  
    Fréchet Inception Distance (FID)
 ############################################   
"""

def get_activations(images, model, batch_size=64, dims=2048):
    n_batches = images.shape[0] // batch_size
    pred_arr = np.empty((n_batches, dims))
    for i in range(n_batches):
        batch = images[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            pred = model(batch)
        pred_arr[i] = pred.numpy()
    return pred_arr

def calculate_fid(real_activations, fake_activations):
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_value

# Calcular FID
def calculate_fid_score(real_images, fake_images):
    # Preprocesar imágenes
    real_images = preprocess_images(real_images)
    fake_images = preprocess_images(fake_images)

    # Obtener activaciones
    real_activations = get_activations(real_images, inception_model)
    fake_activations = get_activations(fake_images, inception_model)

    # Calcular FID
    fid_score = calculate_fid(real_activations, fake_activations)
    return fid_score

"""
NO FUNCIONA EL DATALOADER EL PILLAR UN BATCH Y TAL 

# Cargar las imágenes reales
real_images, _ = next(iter(dataloader))  # Obtener un batch de imágenes reales

# Calcular FID
fake_images = fake_images.numpy()  # Convertir el tensor a un array de NumPy
fid_score = calculate_fid_score(real_images, fake_images)

# Imprimir el resultado
print(f"{'-' * 30}")
print(f"{'Fréchet Inception Distance':^30}")  # Centrar el título
print(f"{'-' * 30}")
print(f"{'FID Score:':<20} {fid_score:.4f}")  # Alinear la etiqueta y mostrar el score
print(f"{'-' * 30}")
"""