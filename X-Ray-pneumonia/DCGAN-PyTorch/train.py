import os,json,torch,random,time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim as optim
import torchvision.utils as vutils
from dcgan import Generator, Discriminator, weights_init
from utils import log_training_info, get_xray  
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
    with open('config.json', 'r') as json_file:
        return json.load(json_file)

def initialize_model(model_type, params, device):
    if model_type == 'dcgan':
        netG = Generator(params).to(device)
        netG.apply(weights_init)
        netD = Discriminator(params).to(device)
        netD.apply(weights_init)
    elif model_type == 'wgan':
        """
        netG = WGANGenerator(params).to(device)
        netG.apply(wgan_weights_init)
        netD = WGANDiscriminator(params).to(device)
        netD.apply(wgan_weights_init)
        """
    return netG, netD

def train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path):
    G_losses, D_losses, img_list = [], [], []
    real_label, fake_label = 0.99, 0.1
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
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Log training info
            if i % 50 == 0:
                log_csv_path = log_training_info('dcgan',epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2)

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

    return G_losses, D_losses, img_list,log_csv_path 

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
                log_csv_path = log_training_info('wgan',epoch, params['nepochs'], i, len(dataloader), errD, errG, D_x, 0, D_G_z2)

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

    return G_losses, D_losses, img_list, log_csv_path 

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
    date = datetime.now().strftime("%Y-%m-%d")
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'params': params
    }, os.path.join(model_path, f'model_ChestCT_{date}.pth'))

    print("==> Modelo final guardado en:", os.path.join(model_path, 'model_ChestCT.pth'))
    return f'model_ChestCT_{date}.pth'



def plot_training_losses(G_losses, D_losses,model, save_dir='evaluation'):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
def main(model_type, params):
    
    device = setup_device()
    config = load_config()
    
    dataloader = get_xray(img_size=params["imsize"], bsize=params["bsize"])
    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    if model_type == 'dcgan':
        print("\033[92mDCGAN model\033[0m")
        model_path = config["model"]["path_dcgan"]
        eval_path = config["model"]["evaluation_dcgan"]
        criterion = torch.nn.BCELoss()

        netG, netD = initialize_model('dcgan', params, device)
        optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Train DCGAN
        G_losses, D_losses, img_list, log_csv_path = train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path)
        finalmodel_name = save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        plot_path = plot_training_losses(G_losses=G_losses, D_losses=D_losses, model=model_type, save_dir=eval_path)
        return finalmodel_name, plot_path, log_csv_path
        

    elif model_type == 'wgan':
        """
        print("\033[92mWGAN model\033[0m")
        model_path = config["model"]["path_wgan"]
        eval_path = config["model"]["evaluation_wgan"]

        netG, netD = initialize_model('wgan', params, device)
        optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
        optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))

        # Entrenar WGAN
        G_losses, D_losses, img_list,log_csv_path = train_wgan(params, dataloader, netG, netD, optimizerG, optimizerD, fixed_noise, device, model_path)
        finalmodel_name = save_model(model_path, netG, netD, optimizerG, optimizerD, params)
        plot_path = plot_training_losses(G_losses=G_losses, D_losses=D_losses, model=model_type, save_dir=eval_path)
        return finalmodel_name, plot_path,log_csv_path"
        """
