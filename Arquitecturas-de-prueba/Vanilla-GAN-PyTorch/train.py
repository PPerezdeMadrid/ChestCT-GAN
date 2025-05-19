import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave, imshow, subplots
from utils import get_chestct
from gan import Generator, Discriminator
import torch.nn as nn
from tqdm import tqdm  

with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path"]
images_path = config["model"]["image_path"]

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")

print(DEVICE, " will be used.\n")

n_noise = 100
D = Discriminator().to(DEVICE)
G = Generator(n_noise).to(DEVICE)

data_loader = get_chestct()

criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999))

max_epoch = params["nepochs"]
step = 0
n_critic = 1

if not os.path.exists(images_path):
    os.makedirs(images_path)

for epoch in tqdm(range(max_epoch), desc="Epochs", dynamic_ncols=True):
    for idx, (images, _) in enumerate(data_loader):
        x = images.to(DEVICE)

        D_labels = torch.ones(x.size(0), 1).to(DEVICE)
        D_fakes = torch.zeros(x.size(0), 1).to(DEVICE)

        D_x_loss = criterion(D(x), D_labels)

        z = torch.randn(x.size(0), n_noise).to(DEVICE)  # Match batch size
        fake_images = G(z)

        D_z_loss = criterion(D(fake_images), D_fakes)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            z = torch.randn(x.size(0), n_noise).to(DEVICE)  # Match batch size
            G_loss = criterion(D(G(z)), D_labels)

            G.zero_grad()
            G_loss.backward()
            G_opt.step()
        
        if step % 500 == 0:
            print(f'Epoch: {epoch}/{max_epoch}, Step: {step}, D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')
        
        if step % 1000 == 0:
            G.eval()
            z = torch.randn(1, n_noise).to(DEVICE)
            img = G(z).detach().cpu().numpy().reshape(64, 64)
            imsave(f'{images_path}/step_{step}.jpg', img, cmap='gray')
            G.train()
        
        step += 1

torch.save({
    'epoch': max_epoch,
    'n_noise': n_noise,
    'D_state_dict': D.state_dict(),
    'G_state_dict': G.state_dict(),
    'D_optimizer_state_dict': D_opt.state_dict(),
    'G_optimizer_state_dict': G_opt.state_dict(),
    'params': params
}, f'../../../model_gan_{params["nepochs"]}')


G.eval()
fig, axs = subplots(3, 3, figsize=(9, 9))  # Create a 3x3 grid of subplots
for i in range(3):
    for j in range(3):
        z = torch.randn(1, n_noise).to(DEVICE)
        img = G(z).detach().cpu().numpy().reshape(64, 64)
        
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].axis('off')  

plt.subplots_adjust(wspace=0, hspace=0)  
plt.show()