import torch
import json
import os
import argparse
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from dcgan import Generator
from datetime import datetime

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path_dcgan"]
image_path = config["model"]["image_path_dcgan"]
real_images_path = config["datasets"]["xray"]  
model_pth = "model_epoch_50.pth"

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default=f'{model_path}/{model_pth}', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=5, help='Number of generated outputs')  # Solo 5 imágenes
args = parser.parse_args()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print(device, " will be used.\n")

state_dict = torch.load(args.load_path, map_location=device)
params = state_dict['params']

netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
netG.eval()

noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((params["imsize"], params["imsize"])),
    transforms.ToTensor()
])

real_dataset = ImageFolder(real_images_path, transform=transform)
real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=int(args.num_output), shuffle=True)
real_img, _ = next(iter(real_loader))

real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())

# Create a grid of images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 5 columns

for i in range(5):
    axes[0, i].imshow(generated_img[i].squeeze(), cmap='gray')
    axes[0, i].set_title("Generated", fontsize=10)
    axes[0, i].axis("off")

for i in range(5):
    axes[1, i].imshow(real_img[i].squeeze(), cmap='gray')
    axes[1, i].set_title("Real", fontsize=10)
    axes[1, i].axis("off")

plt.subplots_adjust(hspace=0.4)

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.suptitle(f"X-Ray real vs generated - {current_date}", fontsize=15)

if not os.path.exists(image_path):
    os.makedirs(image_path)
save_path = os.path.join(image_path, f"XRAY_real_vs_generated_{current_date}.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()
print(f"===> La figura se guardó en: {save_path}")
