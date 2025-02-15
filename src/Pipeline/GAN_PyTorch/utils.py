import torch, csv, json
import torchvision.transforms as transforms
from torchvision import datasets
import os
from datetime import datetime

# Ruta a los datasets
def load_config():
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        return json.load(json_file)

config = load_config()
path = config["datasets"]["chestKaggle"]
train_path = path+"train" 
valid_path = path+"valid"  
test_path = path+"test"  

def get_chestct(img_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(), 
        # transforms.Resize((128, 128)),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
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

def get_NBIA(img_size=64, data_path="Data/Data-Transformed"):
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalización a [-1, 1]
    ])
    # Cargar las imágenes desde la subcarpeta 'cancer'
    dataset = datasets.ImageFolder(root=f'{data_path}', transform=transform)

    # Cargar los datos en un DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Verificar el tamaño del dataset
    print(f"===> Tamaño del conjunto de datos: {len(dataset)}")
    return dataloader



def log_training_info(model, epoch, total_epochs, i, total_iterations, errD, errG, D_x, D_G_z1, D_G_z2):
    """
    Imprime los valores de la iteración actual y guarda la información en un archivo CSV.

    Parameters:
    - epoch: número de la época actual.
    - total_epochs: número total de épocas.
    - i: iteración actual.
    - total_iterations: número total de iteraciones.
    - errD: valor de la pérdida del discriminador.
    - errG: valor de la pérdida del generador.
    - D_x: salida del discriminador en datos reales.
    - D_G_z1: salida del discriminador en datos generados (primera vez).
    - D_G_z2: salida del discriminador en datos generados (segunda vez).
    - log_file: nombre del archivo CSV para guardar los datos.
    """

    fecha = datetime.now().strftime('%Y-%m-%d')
    eval_dir = "evaluation"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    name_csv = f'training_log_{model}_{fecha}.csv'
    save_path = os.path.join(eval, name_csv)
    

    # Imprimir en consola
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
    
    # Modo de escritura dependiendo de la primera iteración (para que sobreescriba)
    mode = 'w' if i == 0 and epoch == 0 else 'a'
    
    with open(save_path, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if mode == 'w':
            csv_writer.writerow(['Epoch', 'Total Epochs', 'Iteration', 'Total Iterations', 'Loss_D', 'Loss_G', 'D(x)', 'D(G(z))_Real', 'D(G(z))_Fake'])
        
        # Escribir los datos de esta iteración
        csv_writer.writerow(log_data)
