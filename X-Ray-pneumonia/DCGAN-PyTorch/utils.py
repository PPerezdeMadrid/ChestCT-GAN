import torch, csv, json
import torchvision.transforms as transforms
from torchvision import datasets
import os
from datetime import datetime
import shutil
import kagglehub
from PIL import Image
import matplotlib.pyplot as plt


# Ruta a los datasets
def load_config():
    # with open('GAN_PyTorch/config.json', 'r') as json_file:
    with open('config.json', 'r') as json_file:
        return json.load(json_file)

config = load_config()
  
def download_xray_data():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)
    return path

def get_xray(img_size=64, bsize=64, data_path=None):
    config = load_config()
    if data_path is None:
        data_path = config["datasets"]["xray"]

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Filtrar imágenes cuyo nombre contenga 'virus'
    def virus_filter(filepath):
        return 'virus' in os.path.basename(filepath).lower() and filepath.lower().endswith(('.png', '.jpg', '.jpeg'))

    dataset = datasets.DatasetFolder(
        root=data_path,
        loader=datasets.folder.default_loader,  
        transform=transform,
        is_valid_file=virus_filter 
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True)

    print(f"===> Tamaño del conjunto de datos: {len(dataset)}")
    images, _ = next(iter(dataloader))
    print(f"===> Tamaño de las imágenes transformadas: {images.shape}")

    # Mostrar 6 imágenes de ejemplo
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(6):
        img = images[i].squeeze().cpu().numpy()  # Convertir a numpy
        img = (img * 0.5) + 0.5  # Desnormalizar
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.show()

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
    eval_dir = f"evaluation/evaluation_{model}"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    name_csv = f'training_log_{model}_{fecha}_64.csv' # Cambiar!!
    save_path = os.path.join(eval_dir, name_csv)
    

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
    return save_path


def prepare_data(source_folder, output_folder):
    """
    Prepara los datos copiando las imágenes de las carpetas NORMAL y PNEUMONIA 
    de train, validate y test en una sola carpeta.

    Args:
    - source_folder (str): Ruta hasta la carpeta chest_xray.
    - output_folder (str): Ruta de la carpeta destino donde se organizarán las imágenes.
    """
    
    source_folder_pneumonia_train = os.path.join(source_folder, 'train', 'PNEUMONIA')
    source_folder_pneumonia_validate = os.path.join(source_folder, 'val', 'PNEUMONIA')
    source_folder_pneumonia_test = os.path.join(source_folder, 'test', 'PNEUMONIA')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pneumonia_folder = os.path.join(output_folder, 'PNEUMONIA')

    if not os.path.exists(pneumonia_folder):
        os.makedirs(pneumonia_folder)

    def copy_images(source_folder, dest_folder):
        for file_name in os.listdir(source_folder):
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)

            if os.path.isfile(source_path):
                shutil.copy(source_path, dest_path)
                print(f"Imagen copiada: {dest_path}")

    print("Copiando imágenes de la clase PNEUMONIA...")
    copy_images(source_folder_pneumonia_train, pneumonia_folder)
    copy_images(source_folder_pneumonia_validate, pneumonia_folder)
    copy_images(source_folder_pneumonia_test, pneumonia_folder)

    print("Proceso completo.")
    return f"{output_folder}/PNEUMONIA"



def get_unique_image_sizes(directory):
    """Recoge todas las imágenes en una carpeta y devuelve los tamaños únicos."""
    if not os.path.isdir(directory):
        print(f"Error: El directorio '{directory}' no existe.")
        return set()
    
    image_sizes = set()
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith(valid_extensions) and os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    image_sizes.add(img.size)  # (ancho, alto)
            except Exception as e:
                print(f"Error al abrir {filename}: {e}")

    print(f"===> Tamaños únicos de imágenes en '{directory}': {image_sizes}")
    return image_sizes

# Ruta hasta chest_xray
# source_folder = '../../../../Dataset_XRAY/chest_xray'

# Ruta de destino para Data_train
# output_folder = '../Data_train/'

# Llamar a la función para preparar los datos
# prepare_data(source_folder, output_folder)

# get_unique_image_sizes('../Data_train/PNEUMONIA')