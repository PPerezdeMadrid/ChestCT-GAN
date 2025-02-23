import argparse

import torch, random, json
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from wgan import Generator

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path"]

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default=f'{model_path}/model_ChestCT.pth', help='Checkpoint to load path from') # modelo previamente entrenado
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Load the checkpoint file.
# Cargar el modelo en CPU si no hay GPU disponible
state_dict = torch.load(args.load_path, map_location=device)


# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
	# Get generated image from the noise vector using
	# the trained generator.
    generated_img = netG(noise).detach().cpu()

# Display the generated image.
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

plt.show()