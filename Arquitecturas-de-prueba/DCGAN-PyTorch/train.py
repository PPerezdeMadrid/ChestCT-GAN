import os,json,torch,random,time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim as optim
import torchvision.utils as vutils
from dcgan512 import Generator as Generator512, Discriminator as Discriminator512, weights_init
from dcgan256 import Generator as Generator256, Discriminator as Discriminator256
from dcgan import Generator, Discriminator, weights_init
# from wgan import Generator as WGANGenerator, Discriminator as WGANDiscriminator, weights_init as wgan_weights_init
from utils import get_chestct, log_training_info, get_NBIA
import argparse
import numpy as np

def setup_device():
    # Use Apple GPU (Metal) if available, else use Intel GPU, else use CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def load_config(config_file):
    with open(config_file, 'r') as json_file:
        return json.load(json_file)

def initialize_model(model_type, params, device):
    if model_type == 'dcgan':
        if params['imsize'] == 512:
            netG = Generator512(params).to(device)
            netG.apply(weights_init)
            netD = Discriminator512(params).to(device)
            netD.apply(weights_init)
        elif params['imsize'] == 64:
            netG = Generator(params).to(device)
            netG.apply(weights_init)
            netD = Discriminator(params).to(device)
            netD.apply(weights_init)
        elif params['imsize'] == 256:
            netG = Generator256(params).to(device)
            netG.apply(weights_init)
            netD = Discriminator256(params).to(device)
            netD.apply(weights_init)
    elif model_type == 'wgan':
        """
        netG = WGANGenerator(params).to(device)
        netG.apply(wgan_weights_init)
        netD = WGANDiscriminator(params).to(device)
        netD.apply(wgan_weights_init)
        """
    return netG, netD

import time
import torch
import torchvision.utils as vutils

def train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path):
    G_losses, D_losses, img_list = [], [], []
    iters = 0

    for epoch in range(params['nepochs']):
        start_time = time.time()
        
        for i, data in enumerate(dataloader, 0):
            real_data = data[0].to(device)
            b_size = real_data.size(0)

            # (1) Update Discriminator (D)
            netD.zero_grad()

            # Etiquetas suavizadas (Técnica de Label Smoothing)
            real_label = 0.85 + torch.rand(1).item() * 0.15  # Entre 0.85 y 1.0
            fake_label = 0.0 + torch.rand(1).item() * 0.15  # Entre 0.0 y 0.15

            # Real batch
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(real_data).view(-1)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Fake batch
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake

            # Evita que D domine demasiado rápido
            if errD.item() > 0.1:
                optimizerD.step()

            # (2) Update Generator (G)
            netG.zero_grad()
            label_real.fill_(real_label)  # Generador intenta engañar a D
            output_fake_G = netD(fake_data).view(-1)
            errG = criterion(output_fake_G, label_real)
            errG.backward()
            D_G_z2 = output_fake_G.mean().item()
            optimizerG.step()

            # Logging
            if i % 50 == 0:
                log_training_info('dcgan', epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2)

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            """
            if (iters % 100 == 0) or (epoch == params['nepochs']-1 and i == len(dataloader)-1):
                with torch.no_grad():
                    fake_data_fixed = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_data_fixed, padding=2, normalize=True))
            """
            iters += 1

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{params['nepochs']}] completada en {epoch_time:.2f} segundos.")

        # Guardar modelo después de cada época
        save_epoch(  
            epoch=epoch + 1,    
            model_path=model_path,
            netG=netG,
            netD=netD,
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            params=params
        )

        if (iters % 100 == 0) or (epoch == params['nepochs']-1 and i == len(dataloader)-1):
                with torch.no_grad():
                    fake_data_fixed = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_data_fixed, padding=2, normalize=True))


    return G_losses, D_losses, img_list

def train_wgan(params, dataloader, netG, netD, optimizerG, optimizerD, fixed_noise, device, model_path):
    # Stores generated images as training progresses
    img_list = []
    # Stores generator losses during training
    G_losses = []
    # Stores discriminator losses during training
    D_losses = []
    
    real_label, fake_label = 1, 0
    iters = 0
    for epoch in range(params['nepochs']):
        start_time = time.time()
        for i, data in enumerate(dataloader, 0):
            real_data = data[0].to(device)
            b_size = real_data.size(0)

            # (1) Update D network
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_data).view(-1)
            errD_real = -output.mean()

            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_data.detach()).view(-1)
            errD_fake = output.mean()

            errD = errD_real + errD_fake
            errD.backward()
            D_x = output.mean().item()
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            output = netD(fake_data).view(-1)
            errG = -output.mean()
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Log training info
            if i % 50 == 0:
                log_training_info('wgan',epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, 0, D_G_z2)

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_data = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

            iters += 1

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}] completed in {epoch_time:.2f} seconds.")

        # Save each epoch
        save_epoch(  
            epoch=epoch + 1,    
            model_path=model_path,
            netG=netG,
            netD=netD,
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            params=params
        )

    return G_losses, D_losses, img_list

def save_epoch(epoch, model_path, netG, netD, optimizerG, optimizerD, params):
    os.makedirs(model_path, exist_ok=True)
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        # }, f'{model_path}/model_epoch_{epoch}_512.pth')
        }, f'{model_path}/model_epoch_{epoch}.pth')


def save_model(model_path, netG, netD, optimizerG, optimizerD, params):
    os.makedirs(model_path, exist_ok=True)
    # Save the final trained model
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'params': params
    # },  f'{model_path}/model_ChestCT_512.pth')
    },  f'{model_path}/model_ChestCT.pth')
    print("==> Final model saved.")


def plot_training_losses(G_losses, D_losses, num_epochs):
    epochs = np.linspace(1, num_epochs, len(G_losses))  # Distribuir puntos en función de los epochs
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(epochs, G_losses, label="Generator Loss (G)")
    plt.plot(epochs, D_losses, label="Discriminator Loss (D)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def save_gif(img_list, filename='ChestTC.gif'):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
    plt.show()
    anim.save(filename, dpi=80, writer='imagemagick')

def main():
    start_time = time.time()
    DATASET_CHOICES = {
        "chestct": get_chestct,
        "nbia": get_NBIA
    }
    parser = argparse.ArgumentParser(description='Train a DCGAN or WGAN model.')
    parser.add_argument('--model', choices=['dcgan', 'wgan'], default='dcgan', help='Choose between "dcgan" and "wgan" models to train.')
    parser.add_argument('--dataset', choices=DATASET_CHOICES.keys(), default='nbia', help='Choose the dataset: "chestct" or "nbia".') # Choose ChestCT to compare architectures
    parser.add_argument('--configFile', type=str, default='config.json', help='Path to JSON config file.')
    args = parser.parse_args()
    print(f"\033[92mUsing the dataset {args.dataset}\033[0m")

    device = setup_device()
    print(f"\033[92mUsing the device {device}\033[0m")
    config = load_config(args.configFile)
    params = config["params"]
    
    # dataloader = get_chestct(params['imsize'])
    # dataloader = DATASET_CHOICES[args.dataset](img_size=params["imsize"])
    dataloader = DATASET_CHOICES[args.dataset](img_size=params["imsize"],bsize=params["bsize"])
    print(f"====> Dataloader {dataloader}")

    sample_batch = next(iter(dataloader))
    
    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
    
    if args.model == 'dcgan':
        print("\033[92mDCGAN model\033[0m")
        model_path = config["model"]["path_dcgan"]
        criterion = torch.nn.BCELoss()

        netG, netD = initialize_model('dcgan', params, device)
        optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Train DCGAN
        G_losses, D_losses, img_list = train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path)
        save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        end_time = time.time()
        print("="*50)
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        # Minutes
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
        # Hours
        print(f"Training completed in {(end_time - start_time) / 3600:.2f} hours.")
        print("="*50)
        fecha = time.strftime("%Y%m%d")
        # save_gif(img_list, f'ChestTC_dcgan_512_{fecha}.gif')
        save_gif(img_list, f'ChestTC_dcgan_{params["imsize"]}_{fecha}.gif')

        
        plot_training_losses(G_losses, D_losses, params['nepochs'])

    elif args.model == 'wgan':
        print("\033[92mWGAN model not available (This for testing)\033[0m")
        """
        model_path = config["model"]["path_wgan"]
        # No need for BCELoss in WGAN
        # Instead, we will calculate the Wasserstein loss directly
        # We will use the gradient penalty in WGAN-GP or simple weight clipping for this implementation

        netG, netD = initialize_model('wgan', params, device)
        optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Train WGAN
        G_losses, D_losses, img_list = train_wgan(params, dataloader, netG, netD, optimizerG, optimizerD, fixed_noise, device, model_path)
        save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        # save_gif(img_list, 'ChestTC_wgan.gif')
        plot_training_losses(G_losses, D_losses, params['num_epochs'])
        """


if __name__ == '__main__':
    main()
