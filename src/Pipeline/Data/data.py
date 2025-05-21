import os
import pydicom
import matplotlib.pyplot as plt

"""
This script is used to explore the dataset of DICOM images.
"""

def get_dicom_files(base_dir):
    """
    Traverses the directory structure and returns a list of DICOM file paths (.dcm).

    Args:
    base_dir (str): Path to the root folder where the images are located.

    Returns:
    list: List of DICOM file paths found.
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
    Displays DICOM images from the given list, arranged in a 3x3 grid.

    Args:
    dicom_files (list): List of DICOM file paths.
    num_images (int): Number of images to display (default 9).
    """
    if not dicom_files:
        print("No DICOM files found.")
        return
    
    num_images = min(num_images, len(dicom_files))

    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        dicom_path = dicom_files[i]
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_subject_images(base_dir, subfolder):
    """
    Displays all DICOM images in a specific subfolder within a base directory.

    Args:
    base_dir (str): Path to the root folder where the images are located.
    subfolder (str): Name of the specific subdirectory (e.g., the name of the subject or study).
    """
    dicom_files = []

    # Traverses the directory structure and looks for DICOM files in the specified subfolder
    for root, dirs, files in os.walk(base_dir):
        if subfolder in root: 
            for file in files:
                if file.endswith(".dcm"):
                    dicom_files.append(os.path.join(root, file))

    if not dicom_files:
        print(f"No se encontraron archivos DICOM en la subcarpeta {subfolder}.")
        return

    print(f"Se encontraron {len(dicom_files)} archivos DICOM en {subfolder}.")

    # Displays DICOM images
    plt.figure(figsize=(15, 15)) 

    for i, dicom_path in enumerate(dicom_files):
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        plt.subplot(5, 5, i + 1)  
        plt.imshow(image, cmap='gray')
        plt.title(f"Imagen {i+1}")
        plt.axis('off')

        if (i + 1) % 25 == 0:  
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, 15))  

    plt.tight_layout()
    plt.show()

def get_dicom_files_recursively(folder_path):
    """
    Recursively searches for DICOM files in a given folder and its subfolders.

    Args:
    folder_path (str): Path of the folder to scan.

    Returns:
    list: List of paths of DICOM files found.
    """
    dicom_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    return dicom_files


def show_subject_images(base_dir, subject_folder):
    """
    Displays all DICOM images within the subfolders of a specific subject,
    including the absolute path to the image.

    Args:
    base_dir (str): Path to the root folder where the images are located.
    subject_folder (str): Name of the subject subdirectory (e.g., 'Lung_Dx-A0003').
    """
    subject_path = os.path.join(base_dir, subject_folder)

    if not os.path.isdir(subject_path):
        print(f"Subject {subject_folder}'s folder do not exist in {base_dir}")
        return

    print(f"Processing images in: {subject_path}")
    
    dicom_files = get_dicom_files_recursively(subject_path)

    if not dicom_files:
        print(f"No DICOM files were found in {subject_folder}.")
        return

    print(f"{len(dicom_files)} DICOM files were found in {subject_folder}.")

    for dicom_path in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_path)
            plt.imshow(ds.pixel_array, cmap='gray')
            plt.title(os.path.abspath(dicom_path), fontsize=8)  
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error reading {dicom_path}: {e}")

def show_subject_subfolder_images(base_dir, subject_folder, subfolder):
    """
    Displays DICOM images from a specific subfolder within a subject folder.

    Args:
    base_dir (str): Path to the root folder where the images are located.
    subject_folder (str): Name of the subject subdirectory (e.g., 'Lung_Dx-A0003').
    subfolder (str): Name of the specific subfolder within the subject.
    """
    subject_path = os.path.join(base_dir, subject_folder)
    subfolder_path = os.path.join(subject_path, subfolder)

    if not os.path.isdir(subfolder_path):
        print(f"The subfolder {subfolder} does not exist in {subject_folder}.")
        return

    print(f"Processing images in: {subfolder_path}")

    dicom_files = get_dicom_files_recursively(subfolder_path)

    if not dicom_files:
        print(f"No DICOM files were found in the subfolder {subfolder}.")
        return

    print(f"{len(dicom_files)} DICOM files were found in the subfolder {subfolder}.")

    # Print the names of the DICOM files
    for dicom_file in dicom_files:
        print(f"Found file: {os.path.basename(dicom_file)}")

    # Display DICOM images
    for i in range(0, len(dicom_files), 25):  # Show 25 images per figure
        plt.figure(figsize=(15, 15))  # New figure

        for j, dicom_path in enumerate(dicom_files[i:i + 25]):
            try:
                ds = pydicom.dcmread(dicom_path)
                plt.subplot(5, 5, j + 1)  # Ensure a valid index
                plt.imshow(ds.pixel_array, cmap='gray')
                
                # Use the filename as the image title
                filename = os.path.basename(dicom_path)
                plt.title(filename, fontsize=8)
                plt.axis('off')
            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")
        
        plt.tight_layout()
        plt.show()

# Configuration of base path and subject
# base_dir = "../../../../ChestCT-NBIA/manifest-1608669183333/Lung-PET-CT-Dx"
# subject_folder = "Lung_Dx-A0004"

# subfolder = "06-10-2006-NA-ThoraxAThoraxRoutine Adult-96697"

# Call the function to display images from the specified subfolder
# show_subject_subfolder_images(base_dir, subject_folder, subfolder)

# Call the function to display images from the subject
# show_subject_images(base_dir, subject_folder)

# Show the first 10 DICOM file paths
# print("First 10 paths:")
# for path in dicom_files[:10]:
    # print(path)

# Show the first 9 DICOM images
# show_dicom_images(dicom_files)

# Show images from a specific subject
# show_subject_images(base_dir, subject_folder)
