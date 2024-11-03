import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from utils import get_chestct  # Asegúrate de que este módulo está definido
from dcgan import weights_init, Generator, Discriminator  # Asegúrate de que este módulo está definido

# Set random seed for reproducibility
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model
params = {
    "bsize": 128,  # Tamaño del batch durante el entrenamiento
    'imsize': 64,  # Tamaño espacial de las imágenes de entrenamiento
    'nc': 1,        # Número de canales en las imágenes de entrenamiento (1 para escala de grises)
    'nz': 100,      # Tamaño del vector latente Z (entrada al generador)
    'ngf': 64,      # Tamaño de los mapas de características en el generador
    'ndf': 64,      # Tamaño de los mapas de características en el discriminador
    'nepochs': 1000,  # Número de épocas de entrenamiento
    'lr': 0.0002,   # Tasa de aprendizaje para los optimizadores
    'beta1': 0.5,   # Beta1 para el optimizador Adam
    'save_epoch': 5  # Intervalo para guardar el modelo
}

# Use GPU if available, else use CPU
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data
dataloader = get_chestct(params)

# Plot the training images
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

# Create the generator
netG = Generator(params).to(device)
netG.apply(weights_init)
print(netG)

# Create the discriminator
netD = Discriminator(params).to(device)
netD.apply(weights_init)
print(netD)

# Binary Cross Entropy loss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses
img_list = []
# Stores generator losses during training
G_losses = []
# Stores discriminator losses during training
D_losses = []

iters = 0

# Create the directory to save models if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

print("Starting Training Loop...")
print("-" * 25)

for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        b_size = real_data.size(0)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # Make accumulated gradients of the discriminator zero
        netD.zero_grad()

        # For real data
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_data).view(-1)  

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors

        # Sample random noise and generate fake data
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)

        # For fake data
        label.fill_(fake_label)
         # Classify all fake batch with D
        output = netD(fake_data.detach()).view(-1)  

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        # Net discriminator loss 
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        # Make accumulated gradients of the generator zero
        netG.zero_grad()

        # We want the fake data to be classified as real
        label.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Check progress of training
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save G's output on a fixed noise
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    # Save the model
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, f'model/model_epoch_{epoch}.pth')

# Save the final trained model
torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
}, 'model/model_ChestCT.pth')



# Plot the training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation showing the improvements of the generator
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('ChestTC.gif', dpi=80, writer='imagemagick')
