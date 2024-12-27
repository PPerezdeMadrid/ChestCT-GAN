import os, json, torch, random
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from utils import get_chestct , log_training_info
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

# No need for BCELoss in WGAN
# Instead, we will calculate the Wasserstein loss directly
# We will use the gradient penalty in WGAN-GP or simple weight clipping for this implementation

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
if not os.path.exists(model_path):
    os.makedirs(model_path)

print("Starting Training Loop...")
print("-" * 25)

for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        b_size = real_data.size(0)

        ############################
        # (1) Update D network: maximize D(x) - D(G(z))
        ###########################

        # Make accumulated gradients of the discriminator zero
        netD.zero_grad()

        # For real data
        output = netD(real_data).view(-1)
        errD_real = -torch.mean(output)  # Wasserstein loss for real data
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors

        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)

        # For fake data
        output = netD(fake_data.detach()).view(-1)
        errD_fake = torch.mean(output)  # Wasserstein loss for fake data
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as the difference between real and fake data
        errD = errD_real + errD_fake
        optimizerD.step()

        # Apply weight clipping (WGAN)
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)  # Clipping the weights

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        # Make accumulated gradients of the generator zero
        netG.zero_grad()

        # We want the fake data to be classified as real
        output = netD(fake_data).view(-1)
        errG = -torch.mean(output)  # Wasserstein loss for generator
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Check progress of training
        if i % 50 == 0:
            log_training_info(epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2)

        # Save the losses for plotting
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
        
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
        }, f'{model_path}/model_epoch_{epoch}.pth')


# Save the final trained model
torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
},  f'{model_path}/model_ChestCT.pth') # FUERA del repo!



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
