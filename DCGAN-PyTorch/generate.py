import torch, random, json, os, argparse
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dcgan import Generator

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path"]
image_path = config["model"]["image_path"]

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default=f'{model_path}/model_ChestCT.pth', help='Checkpoint to load path from') # modelo previamente entrenado
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
state_dict = torch.load(args.load_path, map_location=device)

params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)

# Cargar el modelo previamente entrenado.
netG.load_state_dict(state_dict['generator'])
print(netG)
print(args.num_output)

# Obtener el vector latente Z de una distribución normal estándar.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
	# Get generated image from the noise vector using
	# the trained generator.
    generated_img = netG(noise).detach().cpu()

# Guardar img generadas
if not os.path.exists(image_path):
    os.makedirs(image_path)

for i, img in enumerate(generated_img):
    file_path = os.path.join(image_path, f'generated_image_{i + 1}.png')
    save_image(img, file_path)

# Display the generated image.
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

plt.show()