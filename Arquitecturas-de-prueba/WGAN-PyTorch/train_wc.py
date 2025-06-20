import os, json, torch, random, time  
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
dataloader = get_chestct()

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

"""
We do NOT use Adam as an optimizer for the discriminator in WGAN.
Instead, we use RMSProp with a learning rate of 0.00005.
In the original WGAN paper, the authors recommend using RMSProp for the discriminator:
* “We use RMSProp instead of Adam. We found that Adam sometimes caused the weights to explode with weight clipping, while RMSProp was more stable.”
"""
# optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

optimizerD = optim.RMSprop(netD.parameters(), lr=params['lr'])
optimizerG = optim.RMSprop(netG.parameters(), lr=params['lr'])

img_list = []
G_losses = []
D_losses = []
iters = 0

if not os.path.exists(model_path):
    os.makedirs(model_path)

print("Starting Training Loop...")
print("-" * 25)

total_start_time = time.time()  

for epoch in range(params['nepochs']):
    epoch_start_time = time.time()  
    
    for i, data in enumerate(dataloader, 0):
        real_data = data[0].to(device)
        b_size = real_data.size(0)

         # Update critic
        for _ in range(params['critic_iters']):
            netD.zero_grad()
            output = netD(real_data).view(-1)
            errD_real = -torch.mean(output)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)

            output = netD(fake_data.detach()).view(-1)
            errD_fake = torch.mean(output)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Update the generator once per iter
        netG.zero_grad()
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)  
        fake_data = netG(noise)
        output = netD(fake_data).view(-1)
        errG = -torch.mean(output)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            log_training_info(epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2)

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{params['nepochs']} completed in {epoch_time:.2f} seconds.")

    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, f'{model_path}/model_epoch_{epoch}.pth')


total_time = time.time() - total_start_time
print(f"\nTraining completed in {total_time:.2f} seconds (~{total_time/60:.2f} minutes).\n")

torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
},  f'{model_path}/model_ChestCT.pth')

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('ChestTC.gif', dpi=80, writer='imagemagick')
