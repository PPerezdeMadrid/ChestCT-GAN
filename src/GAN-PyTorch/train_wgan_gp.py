import os,json,torch,random,time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim as optim
import torchvision.utils as vutils
from wgan_gp import Generator as WGANGenerator, Discriminator as WGANDiscriminator, weights_init as wgan_weights_init
from utils import get_chestct, log_training_info
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

def load_config():
    with open('config.json', 'r') as json_file:
        return json.load(json_file)

def initialize_model(params, device):
    netG = WGANGenerator(params).to(device)
    netG.apply(wgan_weights_init)
    netD = WGANDiscriminator(params).to(device)
    netD.apply(wgan_weights_init)
    return netG, netD


def main():
    # Load configuration
    config = load_config()
    params = config["params"]

    # Set device
    device = setup_device()
    seed= 42

    # Initialize random seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    dataloader = get_chestct()

    # Initialize models
    netG = initialize_model(params, device)

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999))

    # Training loop
    for epoch in range(config["num_epochs"]):
        for i, data in enumerate(dataloader, 0):
            real_images = data[0].to(device)
            
            # Train Generator
            netG.zero_grad()
            noise = torch.randn(config["batch_size"], 100, 1, 1, device=device)
            fake_images = netG(noise)
            g_loss = -torch.mean(fake_images)  # Placeholder for loss computation
            g_loss.backward()
            optimizerG.step()

            # Logging
            if i % 50 == 0:
                print(f"[{epoch}/{config['num_epochs']}][{i}/{len(dataloader)}] Loss_G: {g_loss.item()}")
        
        # Save generator every epoch
        torch.save(netG.state_dict(), os.path.join(config["output_path"], f"generator_epoch_{epoch}.pth"))
        print(f"Generator saved for epoch {epoch}")

    # Save final generator model
    torch.save(netG.state_dict(), os.path.join(config["output_path"], "model.pth"))
    print("Final generator model saved.")

if __name__ == "__main__":
    main()