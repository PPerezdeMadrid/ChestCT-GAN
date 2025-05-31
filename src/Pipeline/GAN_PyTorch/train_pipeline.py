import os,json,torch,random,time, logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim as optim
import torchvision.utils as vutils
from GAN_PyTorch.dcgan import Generator, Discriminator, weights_init
from GAN_PyTorch.wgan import Generator as WGANGenerator, Discriminator as WGANDiscriminator, weights_init as wgan_weights_init
from GAN_PyTorch.utils import get_chestct, log_training_info, get_NBIA
from datetime import datetime

def setup_device():
    # Use Apple GPU (Metal) if available, else use Intel GPU, else use CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)

def initialize_model(model_type, params, device):
    if model_type == 'dcgan':
        netG = Generator(params).to(device)
        netG.apply(weights_init)
        netD = Discriminator(params).to(device)
        netD.apply(weights_init)
    elif model_type == 'wgan':
        netG = WGANGenerator(params).to(device)
        netG.apply(wgan_weights_init)
        netD = WGANDiscriminator(params).to(device)
        netD.apply(wgan_weights_init)
    return netG, netD

def train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path, date):
    G_losses, D_losses, img_list = [], [], []
    iters = 0
    name_csv = f'training_log_dcgan_{date}.csv'

    for epoch in range(params['nepochs']):
        start_time = time.time()
        
        for i, data in enumerate(dataloader, 0):
            real_data = data[0].to(device)
            b_size = real_data.size(0)

            # (1) Update Discriminator (D)
            netD.zero_grad()

            # Label smoothing
            real_label = 0.85 + torch.rand(1).item() * 0.15  # Between 0.85 and 1.0
            fake_label = 0.0 + torch.rand(1).item() * 0.15  # Between 0.0 and 0.15

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

            # Avoid overfitting by limiting the number of updates to D
            if errD.item() > 0.1:
                optimizerD.step()

            # (2) Update Generator (G)
            netG.zero_grad()
            label_real.fill_(real_label)  # Generator tries to make D believe that the fake data is real
            output_fake_G = netD(fake_data).view(-1)
            errG = criterion(output_fake_G, label_real)
            errG.backward()
            D_G_z2 = output_fake_G.mean().item()
            optimizerG.step()

            # Logging
            if i % 50 == 0:
                log_training_info('dcgan', epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2, name_csv)

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        epoch_time = time.time() - start_time
        logging.info(f"Epoch [{epoch + 1}/{params['nepochs']}] completada en {epoch_time:.2f} segundos.")

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

    return G_losses, D_losses, img_list, name_csv

def train_wgan(params, dataloader, netG, netD, optimizerG, optimizerD, fixed_noise, device, model_path, date):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    total_start_time = time.time()
    name_csv = f'training_log_wgan_{date}.csv'

    print("Starting WGAN Training Loop...\n" + "-"*30)

    for epoch in range(params['nepochs']):
        epoch_start_time = time.time()

        for i, data in enumerate(dataloader, 0):
            real_data = data[0].to(device)
            b_size = real_data.size(0)

            # Update critic
            for _ in range(params['critic_iters']):
                netD.zero_grad()

                # Real
                output_real = netD(real_data).view(-1)
                errD_real = -torch.mean(output_real)
                errD_real.backward()
                D_x = output_real.mean().item()

                # Fake
                noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
                fake_data = netG(noise)
                output_fake = netD(fake_data.detach()).view(-1)
                errD_fake = torch.mean(output_fake)
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()

                # Total loss and step
                errD = errD_real + errD_fake
                optimizerD.step()

                # Clipping weights
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # Update generator
            netG.zero_grad()
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            output = netD(fake_data).view(-1)
            errG = -torch.mean(output)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Logging
            if i % 50 == 0:
                log_training_info(epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2, name_csv)

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 100 == 0) or ((epoch == params['nepochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_images = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

            iters += 1

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{params['nepochs']} completed in {epoch_time:.2f} seconds.")

        # Save model every few epochs
        save_epoch(epoch, model_path, netG, netD, optimizerG, optimizerD, params)

    total_time = time.time() - total_start_time
    print(f"\nWGAN training completed in {total_time:.2f} seconds (~{total_time/60:.2f} minutes).\n")

    return G_losses, D_losses, img_list, name_csv

def save_epoch(epoch, model_path, netG, netD, optimizerG, optimizerD, params):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"==> Carpeta creada: {model_path}")
        
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, f'{model_path}/model_epoch_{epoch}.pth')


def save_model(model_path, netG, netD, optimizerG, optimizerD, params):
    date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    model_filename = f'model_ChestCT_{date}.pth'
    full_path = os.path.join(model_path, model_filename)

    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'params': params
    }, full_path)

    print(f"==> Modelo final guardado en: {full_path}")
    return model_filename



def plot_training_losses(G_losses, D_losses,model, save_dir='evaluation'):
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    save_path = f'{save_dir}/training_losses_{current_time}_{model}.png'

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.close() 

    print(f"Training losses plot saved as: {save_path}")
    return save_path


# En train.py
def main(arg, params):
    DATASET_CHOICES = {
        "chestct": get_chestct,
        "nbia": get_NBIA
    }
    
    date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    model_type = arg['model_type']
    dataset = arg['dataset']
    
    device = setup_device()
    config = load_config()
    
    dataloader = DATASET_CHOICES[dataset](img_size=params["imsize"])
    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    if model_type == 'dcgan':
        print("\033[92mDCGAN model\033[0m")
        model_path = config["model"]["path_dcgan"]
        eval_path = config["model"]["evaluation_dcgan"]
        criterion = torch.nn.BCELoss()

        netG, netD = initialize_model('dcgan', params, device)
        optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Entrenar DCGAN
        G_losses, D_losses, img_list, log_csv_name = train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path, date)
        finalmodel_name = save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        plot_path = plot_training_losses(G_losses=G_losses, D_losses=D_losses, model=model_type, save_dir=eval_path)
        return finalmodel_name, plot_path, log_csv_name
        

    elif model_type == 'wgan':
        print("\033[92mWGAN model\033[0m")
        model_path = config["model"]["path_wgan"]
        eval_path = config["model"]["evaluation_wgan"]

        netG, netD = initialize_model('wgan', params, device)
        optimizerD = optim.RMSprop(netD.parameters(), lr=params['lr'])
        optimizerG = optim.RMSprop(netG.parameters(), lr=params['lr'])
        fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)


        # Entrenar WGAN
        G_losses, D_losses, img_list,log_csv_path = train_wgan(params, dataloader, netG, netD, optimizerG, optimizerD, fixed_noise, device, model_path, date)
        finalmodel_name = save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        plot_path = plot_training_losses(G_losses=G_losses, D_losses=D_losses, model=model_type, save_dir=eval_path)
        return finalmodel_name, plot_path,log_csv_path
