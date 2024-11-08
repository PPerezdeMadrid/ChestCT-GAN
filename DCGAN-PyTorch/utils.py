import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

# Ruta a los datasets
path="../../../ChestCTKaggle/Data/"
train_path = path+"train" 
valid_path = path+"valid"  
test_path = path+"test"  

def get_chestct(params):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        # transforms.Resize((128, 128)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalización a [-1, 1]
    ])

    # Cargar los datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # Verificar el tamaño del dataset
    print(f"===> Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"===> Tamaño del conjunto de validación: {len(valid_dataset)}")
    print(f"===> Tamaño del conjunto de prueba: {len(test_dataset)}")

    # Concatenar los datasets
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset, test_dataset])

    # Crear un solo data loader
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True)

    # Verificar el tamaño del dataset combinado
    print(f"===> Tamaño del conjunto combinado: {len(combined_dataset)}")

    return combined_loader
