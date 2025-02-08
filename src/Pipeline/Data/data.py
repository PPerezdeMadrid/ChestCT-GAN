import os
import pydicom
import matplotlib.pyplot as plt

# pip install --break-system-packages -r requirements.txt

def get_dicom_files(base_dir):
    """
    Recorre la estructura de directorios y devuelve una lista con las rutas de los archivos DICOM (.dcm).
    
    Args:
        base_dir (str): Ruta a la carpeta raíz donde se encuentran las imágenes.

    Returns:
        list: Lista de rutas de archivos DICOM encontrados.
    """
    dicom_files = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))

    print(f"Se encontraron {len(dicom_files)} archivos DICOM.")
    return dicom_files


def show_dicom_images(dicom_files, num_images=9):
    """
    Muestra imágenes DICOM de la lista dada, organizadas en una cuadrícula de 3x3.

    Args:
        dicom_files (list): Lista de rutas de archivos DICOM.
        num_images (int): Número de imágenes a mostrar (por defecto 9).
    """
    if not dicom_files:
        print("No se encontraron archivos DICOM.")
        return
    
    num_images = min(num_images, len(dicom_files))

    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        dicom_path = dicom_files[i]
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Imagen {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_subject_images(base_dir, subfolder):
    """
    Muestra todas las imágenes DICOM de una subcarpeta específica dentro de un directorio base.

    Args:
        base_dir (str): Ruta a la carpeta raíz donde se encuentran las imágenes.
        subfolder (str): Nombre del subdirectorio específico (por ejemplo, el nombre del sujeto o estudio).
    """
    dicom_files = []

    # Recorre la estructura de directorios y busca archivos DICOM en la subcarpeta especificada
    for root, dirs, files in os.walk(base_dir):
        if subfolder in root:  # Filtra por la subcarpeta especificada
            for file in files:
                if file.endswith(".dcm"):
                    dicom_files.append(os.path.join(root, file))

    if not dicom_files:
        print(f"No se encontraron archivos DICOM en la subcarpeta {subfolder}.")
        return

    print(f"Se encontraron {len(dicom_files)} archivos DICOM en {subfolder}.")

    # Muestra las imágenes DICOM
    plt.figure(figsize=(15, 15))  # Ajusta el tamaño de la figura según sea necesario

    for i, dicom_path in enumerate(dicom_files):
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        plt.subplot(5, 5, i + 1)  # Ajusta las filas y columnas de la cuadrícula según el número de imágenes
        plt.imshow(image, cmap='gray')
        plt.title(f"Imagen {i+1}")
        plt.axis('off')

        if (i + 1) % 25 == 0:  # Muestra 25 imágenes por página
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, 15))  # Nueva figura para las siguientes imágenes

    plt.tight_layout()
    plt.show()

def get_dicom_files_recursively(folder_path):
    """
    Busca recursivamente archivos DICOM en una carpeta dada y sus subcarpetas.

    Args:
        folder_path (str): Ruta de la carpeta a escanear.

    Returns:
        list: Lista de rutas de archivos DICOM encontrados.
    """
    dicom_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    return dicom_files


def show_subject_images(base_dir, subject_folder):
    """
    Muestra todas las imágenes DICOM dentro de las subcarpetas de un sujeto específico,
    incluyendo la ruta absoluta en la imagen.

    Args:
        base_dir (str): Ruta a la carpeta raíz donde se encuentran las imágenes.
        subject_folder (str): Nombre del subdirectorio del sujeto (por ejemplo, 'Lung_Dx-A0003').
    """
    subject_path = os.path.join(base_dir, subject_folder)

    if not os.path.isdir(subject_path):
        print(f"La carpeta del sujeto {subject_folder} no existe en {base_dir}")
        return

    print(f"Procesando imágenes en: {subject_path}")
    
    dicom_files = get_dicom_files_recursively(subject_path)

    if not dicom_files:
        print(f"No se encontraron archivos DICOM en {subject_folder}.")
        return

    print(f"Se encontraron {len(dicom_files)} archivos DICOM en {subject_folder}.")

    for dicom_path in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_path)
            plt.imshow(ds.pixel_array, cmap='gray')
            plt.title(os.path.abspath(dicom_path), fontsize=8)  # Mostrar el path en la imagen
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error leyendo {dicom_path}: {e}")

def show_subject_subfolder_images(base_dir, subject_folder, subfolder):
    """
    Muestra imágenes DICOM de una subcarpeta específica dentro de la carpeta de un sujeto.

    Args:
        base_dir (str): Ruta a la carpeta raíz donde se encuentran las imágenes.
        subject_folder (str): Nombre del subdirectorio del sujeto (por ejemplo, 'Lung_Dx-A0003').
        subfolder (str): Nombre de la subcarpeta específica dentro del sujeto.
    """
    subject_path = os.path.join(base_dir, subject_folder)
    subfolder_path = os.path.join(subject_path, subfolder)

    if not os.path.isdir(subfolder_path):
        print(f"La subcarpeta {subfolder} no existe en {subject_folder}.")
        return

    print(f"Procesando imágenes en: {subfolder_path}")

    dicom_files = get_dicom_files_recursively(subfolder_path)

    if not dicom_files:
        print(f"No se encontraron archivos DICOM en la subcarpeta {subfolder}.")
        return

    print(f"Se encontraron {len(dicom_files)} archivos DICOM en la subcarpeta {subfolder}.")

    # Imprimir nombres de los archivos DICOM
    for dicom_file in dicom_files:
        print(f"Archivo encontrado: {os.path.basename(dicom_file)}")

    # Mostrar imágenes DICOM
    for i in range(0, len(dicom_files), 25):  # Muestra 25 imágenes por figura
        plt.figure(figsize=(15, 15))  # Nueva figura

        for j, dicom_path in enumerate(dicom_files[i:i + 25]):
            try:
                ds = pydicom.dcmread(dicom_path)
                plt.subplot(5, 5, j + 1)  # Asegurar un índice válido
                plt.imshow(ds.pixel_array, cmap='gray')
                
                # Usar el nombre del archivo como título de la imagen
                filename = os.path.basename(dicom_path)
                plt.title(filename, fontsize=8)
                plt.axis('off')
            except Exception as e:
                print(f"Error leyendo {dicom_path}: {e}")
        
        plt.tight_layout()
        plt.show()

# Configuración de la ruta base y el sujeto
base_dir = "../../../../ChestCT-NBIA/manifest-1608669183333/Lung-PET-CT-Dx"
subject_folder = "Lung_Dx-A0004"

subfolder = "06-10-2006-NA-ThoraxAThoraxRoutine Adult-96697"

# Llamada a la función para mostrar imágenes de la subcarpeta especificada
show_subject_subfolder_images(base_dir, subject_folder, subfolder)

# Llamada a la función para mostrar imágenes del sujeto
# show_subject_images(base_dir, subject_folder)


# Mostrar las primeras 10 rutas de archivos DICOM
# print("Primeras 10 rutas:")
# for path in dicom_files[:10]:
    # print(path)

# Mostrar las primeras 9 imágenes DICOM
# show_dicom_images(dicom_files)

# Mostrar imágenes de un sujeto en concreto
# show_subject_images(base_dir,subject_folder)





