import torch, csv
import torchvision.transforms as transforms
from torchvision import datasets
import os

"""
This implementation was made to compare different architectures.
As this is the most basic GAN, some parameters are hardcoded.
"""
# Note: The path to the dataset is hardcoded here. You may want to change it to a relative path or use a config file.
path="../../../../TFG/ChestCTKaggle/Data/"
train_path = path+"train" 
valid_path = path+"valid"  
test_path = path+"test"  

def get_chestct():
    transform = transforms.Compose([
        transforms.Grayscale(),  
        transforms.Resize((64,64)),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    print(f"===> Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"===> Tamaño del conjunto de validación: {len(valid_dataset)}")
    print(f"===> Tamaño del conjunto de prueba: {len(test_dataset)}")

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset, test_dataset])

    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=128, shuffle=True, drop_last=True)

    print(f"===> Tamaño del conjunto combinado: {len(combined_dataset)}")

    return combined_loader


def log_training_info(epoch, total_epochs, i, total_iterations, errD, errG, D_x, D_G_z1, D_G_z2, log_file='training_log_gan.csv'):
    """
    Prints the values ​​for the current iteration and saves the information to a CSV file.

    Parameters:
    - epoch: Current epoch number.
    - total_epochs: Total number of epochs.
    - i: Current iteration.
    - total_iterations: Total number of iterations.
    - errD: Discriminator loss value.
    - errG: Generator loss value.
    - D_x: Discriminator output on real data.
    - D_G_z1: Discriminator output on generated data (first time).
    - D_G_z2: Discriminator output on generated data (second time).
    - log_file: Name of the CSV file to save the data to.
    """
 
    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
        epoch, total_epochs, i, total_iterations,
        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
    log_data = [
        epoch, 
        total_epochs, 
        i, 
        total_iterations, 
        errD.item(), 
        errG.item(), 
        D_x, 
        D_G_z1, 
        D_G_z2
    ]
    
    mode = 'w' if i == 0 and epoch == 0 else 'a'
    
    with open(log_file, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if mode == 'w':
            csv_writer.writerow(['Epoch', 'Total Epochs', 'Iteration', 'Total Iterations', 'Loss_D', 'Loss_G', 'D(x)', 'D(G(z))_Real', 'D(G(z))_Fake'])
        csv_writer.writerow(log_data)


