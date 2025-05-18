import os, json, torch, random
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time  # Importamos la librería para medir el tiempo

from utils import get_chestct, log_training_info
from wgan import weights_init, Generator, Discriminator

# Set random seed for reproducibility
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Upload parameters and model_path
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

params = config["params"]
model_path = config["model"]["path"]

# Use GPU or MPS if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device, " will be used.\n")

# Get the data
dataloader = get_chestct()

# Show some training images
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# Models
netG = Generator(params).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(params).to(device)
netD.apply(weights_init)
print(netD)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Gradient penalty function
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training loop
print("Starting Training Loop...")
print("-" * 25)

lambda_gp = 10  # WGAN-GP gradient penalty coefficient

total_start_time = time.time()

for epoch in range(params['nepochs']):
    start_time = time.time()
    
    for i, data in enumerate(dataloader, 0):
        real_data = data[0].to(device)
        b_size = real_data.size(0)

        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()

        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)

        real_validity = netD(real_data)
        fake_validity = netD(fake_data.detach())
        gp = compute_gradient_penalty(netD, real_data.data, fake_data.data)

        errD = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        # if i % params.get("critic_iters", 5) == 0: # Update G every 5 iterations
        if i % params['critic_iters'] == 0:
            netG.zero_grad()
            fake_data = netG(noise)
            output = netD(fake_data)
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()
            G_losses.append(errG.item())

        D_losses.append(errD.item())

        if i % 50 == 0:
            log_training_info(epoch, params['nepochs'], i, len(dataloader), errD, errG, 
                              real_validity.mean().item(), fake_validity.mean().item(), output.mean().item())

        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{params['nepochs']}] completed in {epoch_time:.2f} seconds.")

    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, f'{model_path}/model_epoch_{epoch}.pth')

# Save final model
torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
},  f'{model_path}/model_ChestCT.pth')

# Calculate and display total training time
total_time = time.time() - total_start_time
total_seconds = total_time
total_minutes = total_time / 60
total_hours = total_time / 3600

print("\nEntrenamiento completo en:")
print(f"{total_seconds:.2f} segundos")
print(f"{total_minutes:.2f} minutos")
print(f"{total_hours:.2f} horas")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('ChestTC.gif', dpi=80, writer='imagemagick')
