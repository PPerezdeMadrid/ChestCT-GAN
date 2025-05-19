import torch, random, json, os, argparse
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Generator
from dcgan512 import Generator as Generator512
from dcgan256 import Generator as Generator256
from PIL import Image

# config_path = 'config.json'
config_path = '1stHiperparams.json'
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = "model_prueba/model_dcgan/"
image_path = f"{config["model"]["image_path_dcgan"]}/generated_{params["imsize"]}"
real_image_path = f"{config["datasets"]["nbia"]}/cancer"

parser = argparse.ArgumentParser()
model_name = "model_ChestCT.pth"
parser.add_argument('-load_path', default=f'{model_path}/{model_name}', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
parser.add_argument('-compare', action='store_true', help='Show comparison between generated and real images')
args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print(device, " will be used.\n")

state_dict = torch.load(args.load_path, map_location=device)
params = state_dict['params']

if params['imsize'] == 64:
    netG = Generator(params).to(device)
    netG.load_state_dict(state_dict['generator'])
elif params['imsize'] == 512:
    netG = Generator512(params).to(device)
    netG.load_state_dict(state_dict['generator'])
elif params['imsize'] == 256:
    netG = Generator256(params).to(device)
    netG.load_state_dict(state_dict['generator'])
print(netG)
print(args.num_output)

noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

if not os.path.exists(image_path):
    os.makedirs(image_path)

generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())

for i, img in enumerate(generated_img):
    file_path = os.path.join(image_path, f'generated_image_{i + 1}.png')
    vutils.save_image(img, file_path)

grid = vutils.make_grid(generated_img, padding=2, normalize=True)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(grid, (1, 2, 0)))

save_path = os.path.join(image_path, "generated_grid.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()

# --------------------------
# COMPARACIÓN CON REALES
# --------------------------
if args.compare:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Generated vs Real Images", fontsize=16)

    # Mostrar 4 imágenes generadas
    for i in range(4):
        gen_img = generated_img[i].squeeze().numpy()
        axes[0, i].imshow(gen_img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Generated {i+1}')

    # Mostrar 4 imágenes reales desde la carpeta
    real_images = sorted([f for f in os.listdir(real_image_path) if f.endswith('.png')])[:4]
    for i, filename in enumerate(real_images):
        real_img = Image.open(os.path.join(real_image_path, filename)).convert('L')
        axes[1, i].imshow(np.array(real_img), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Real {i+1}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    compare_path = os.path.join(image_path, "comparison_grid.png")
    plt.savefig(compare_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
