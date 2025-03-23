import torch, random, json, os, argparse
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Generator

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path_dcgan"]
image_path = config["model"]["image_path_dcgan"]

parser = argparse.ArgumentParser()
model_name = "model_epoch_100.pth"
parser.add_argument('-load_path', default=f'{model_path}/{model_name}', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print(device, " will be used.\n")

state_dict = torch.load(args.load_path, map_location=device)
params = state_dict['params']

# Create generator
netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
print(netG)
print(args.num_output)

# Generate noise and images
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

# Ensure output directory exists
if not os.path.exists(image_path):
    os.makedirs(image_path)

# Normalize images before saving
generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())

for i, img in enumerate(generated_img):
    file_path = os.path.join(image_path, f'generated_image_{i + 1}.png')
    vutils.save_image(img, file_path)

# Save the grid image as it is displayed
grid = vutils.make_grid(generated_img, padding=2, normalize=True)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(grid, (1, 2, 0)))

# Save the figure
save_path = os.path.join(image_path, "generated_grid.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()
