import torch, csv, json
import torchvision.transforms as transforms
from torchvision import datasets
import os
from datetime import datetime

"""
If you are using the datset from Kaggle, you will be using by default InitialConfig.json
As InitialConfig.json is the config file used to compare the different architectures.
"""
def load_config():
    with open('InitialConfig.json', 'r') as json_file: # You can change this to your config file
        return json.load(json_file)

config = load_config()
path = config["datasets"]["chestKaggle"]
train_path = path+"train" 
valid_path = path+"valid"  
test_path = path+"test"  

def get_dataloader(dataset_type, img_size=64):
    if dataset_type == 'chestct':
        return get_chestct(img_size)
    elif dataset_type == 'nbia':
        return get_NBIA(img_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
def get_chestct(img_size=64, bsize=128):
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # Verify the size of the dataset
    print(f"===> Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"===> Tamaño del conjunto de validación: {len(valid_dataset)}")
    print(f"===> Tamaño del conjunto de prueba: {len(test_dataset)}")

    # Concatenate the datasets
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset, test_dataset])
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=bsize, shuffle=True)

    print(f"===> Tamaño del conjunto combinado: {len(combined_dataset)}")

    return combined_loader

def get_NBIA(img_size=512, bsize=8):
    config = load_config()
    data_path = config["datasets"]["nbia"]
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    # Load images from the 'cancer' subfolder
    dataset = datasets.ImageFolder(root=f'{data_path}', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True)
    
    print(f"===> Tamaño del conjunto de datos: {len(dataset)}")
    
    images, _ = next(iter(dataloader))

    return dataloader



def log_training_info(model, epoch, total_epochs, i, total_iterations, errD, errG, D_x, D_G_z1, D_G_z2):
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

    date = datetime.now().strftime('%Y-%m-%d')
    eval_dir = f"evaluation_prueba/evaluation_{model}"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    name_csv = f'training_log_{model}_{date}.csv'
    save_path = os.path.join(eval_dir, name_csv)
    

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
    
    # writing mode depending on the first iteration (to overwrite)
    mode = 'w' if i == 0 and epoch == 0 else 'a'
    
    with open(save_path, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if mode == 'w':
            csv_writer.writerow(['Epoch', 'Total Epochs', 'Iteration', 'Total Iterations', 'Loss_D', 'Loss_G', 'D(x)', 'D(G(z))_Real', 'D(G(z))_Fake'])
    
        csv_writer.writerow(log_data)
    return save_path
