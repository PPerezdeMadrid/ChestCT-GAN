import os
import time
import pydicom
import torch
import logging
import lpips
from PIL import Image, ImageEnhance
from torchvision import transforms
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
This script generate the dataset.
"""

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array.astype(float)

    # Normalizing the image to 8 bits (0-255)
    image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    image_array = np.uint8(image_array)

    # Save the image as PNG
    img = Image.fromarray(image_array)
    img.save(output_path)

def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

def calculate_lpips(model, img1_path, img2_path, device):
    img1 = load_image(img1_path, device)
    img2 = load_image(img2_path, device)
    with torch.no_grad():
        lpips_score = model(img1, img2)
    return lpips_score.item()

def calculate_psnr(image1, image2):
    # Convert images to grayscale if they are in color
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size if they are not
    if image1_gray.shape != image2_gray.shape:
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))

    # Calculate PSNR
    score = psnr(image1_gray, image2_gray)
    return score

def setup_logging():
    log_filename = f"data_generated_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

"""
This is an alternative function not implemented in the project.
# Future Lines 
# Processes DICOM folders by converting images to PNG and filtering them using LPIPS and PSNR scores 
# against both reference and discarded image sets to classify them into transformed or discarded folders.

"""
def process_dicom_folders_lpips_psnr(path_NBIA_Data, reference_images_paths, discarded_reference_images_paths, transformed_dir, discarded_dir, threshold_lpips=0.3500, threshold_psnr=20.0, threshold_discard_lpips=0.3500, threshold_discard_psnr=20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = lpips.LPIPS(net='vgg').to(device)

    ensure_directory_exists(transformed_dir)
    ensure_directory_exists(discarded_dir)

    results = []
    image_counter = 1

    metadata_path = os.path.join(path_NBIA_Data, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    metadata = metadata[metadata['Study UID'].isin(valid_study_descriptions)]

    logging.info(f"Found {len(metadata)} records with valid Study UID.")
    start_time = time.time()

    for _, row in metadata.iterrows():
        folder_path = os.path.join(path_NBIA_Data, row['File Location'])
        if not os.path.exists(folder_path):
            logging.info(f"Folder not found: {folder_path}")
            continue

        logging.info(f"Processing folder: {folder_path}")

        for root, _, files in os.walk(folder_path):
            # Filtrar los archivos DICOM
            dicom_files = sorted([f for f in files if f.lower().endswith('.dcm')])

            # Excluir los primeros 4 y últimos 4 archivos
            dicom_files = dicom_files[4:-4]

            for file in dicom_files:
                dicom_path = os.path.join(root, file)
                png_filename = f"ChestCT_{image_counter}.png"
                output_png_path = os.path.join(transformed_dir, png_filename)

                dicom_to_png(dicom_path, output_png_path)

                move_to_transformed = False
                move_to_discarded = False
                best_lpips = None
                best_psnr = None
                best_lpips_discard = None
                best_psnr_discard = None

                for reference_image_path in reference_images_paths:
                    lpips_value = calculate_lpips(model, reference_image_path, output_png_path, device)

                    if lpips_value <= threshold_lpips:
                        ref_image = cv2.imread(reference_image_path)
                        test_image = cv2.imread(output_png_path)
                        psnr_value = calculate_psnr(ref_image, test_image)
                        logging.info(f"PSNR (transf) {png_filename} vs {reference_image_path}: {psnr_value:.4f}")
                        best_lpips = lpips_value
                        best_psnr = psnr_value
                        if psnr_value >= threshold_psnr:
                            move_to_transformed = True
                            break
                    else:
                        best_lpips = lpips_value

                for discarded_reference_image_path in discarded_reference_images_paths:
                    lpips_value_discarded = calculate_lpips(model, discarded_reference_image_path, output_png_path, device)

                    if lpips_value_discarded <= threshold_discard_lpips:
                        discarded_ref_image = cv2.imread(discarded_reference_image_path)
                        test_image = cv2.imread(output_png_path)
                        psnr_value_discarded = calculate_psnr(discarded_ref_image, test_image)
                        logging.info(f"PSNR (discard) {png_filename} vs {discarded_reference_image_path}: {psnr_value_discarded:.4f}")
                        best_lpips_discard = lpips_value_discarded
                        best_psnr_discard = psnr_value_discarded
                        if psnr_value_discarded >= threshold_discard_psnr:
                            move_to_discarded = True
                            break
                    else:
                        best_lpips_discard = lpips_value_discarded

                if move_to_transformed:
                    final_path = os.path.join(transformed_dir, png_filename)
                    logging.info(f"✅ {png_filename} moved to TRANSFORMED")
                elif move_to_discarded:
                    final_path = os.path.join(discarded_dir, png_filename)
                    logging.info(f"❌ {png_filename} moved to DISCARDED (discard refs)")
                else:
                    final_path = os.path.join(discarded_dir, png_filename)
                    logging.info(f"⚠️ {png_filename} moved to DISCARDED (default)")

                os.rename(output_png_path, final_path)

                results.append((png_filename,
                                best_lpips if best_lpips is not None else -1,
                                best_psnr if best_psnr is not None else -1,
                                best_lpips_discard if best_lpips_discard is not None else -1,
                                best_psnr_discard if best_psnr_discard is not None else -1))
                image_counter += 1

    with open("lpips_psnr_results.csv", "w") as f:
        f.write("image_name,lpips_score,psnr_score,discarded_lpips_score,discarded_psnr_score\n")
        for img_name, lpips_score, psnr_score, discarded_lpips_score, discarded_psnr_score in results:
            f.write(f"{img_name},{lpips_score:.4f},{psnr_score:.4f},{discarded_lpips_score:.4f},{discarded_psnr_score:.4f}\n")

    logging.info("Processing completed. Results saved to lpips_psnr_results.csv")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[33mExecution Time: {elapsed_time:.2f} seconds\033[0m")
    logging.info(f"\033[33mProcessing completed. Execution Time: {elapsed_time:.2f} seconds\033[0m")



def process_dicom_folders(metadata_csv_path, dicom_root_dir, reference_images_paths, transformed_dir, discarded_dir, threshold=0.3500):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = lpips.LPIPS(net='alex').to(device)
    
    ensure_directory_exists(transformed_dir)
    ensure_directory_exists(discarded_dir)

    results = []
    image_counter = 1  # Counter for naming images as ChestCT_X.png

    # Read the metadata CSV
    metadata = pd.read_csv(metadata_csv_path)
    
    # Filter rows with valid Study Description
    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    metadata = metadata[metadata['Study UID'].isin(valid_study_descriptions)]
    
    print(f"Found {len(metadata)} records with valid Study UID.")
    start_time = time.time()

    # Iterate over metadata rows with a progress bar
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing folders"):
        # Get the folder path from File Location
        folder_path = os.path.join(dicom_root_dir, row['File Location'])
        
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
        
        # print(f"Processing folder: {folder_path}")
        
        # Walk through DICOM images in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    png_filename = f"ChestCT_{image_counter}.png"
                    output_png_path = os.path.join(transformed_dir, png_filename)
                    
                    dicom_to_png(dicom_path, output_png_path)
                    
                    move_to_transformed = False
                    
                    for reference_image_path in reference_images_paths:
                        lpips_value = calculate_lpips(model, reference_image_path, output_png_path, device)
                        logging.info(f"LPIPS score for {png_filename} with {reference_image_path}: {lpips_value:.4f}")
                        
                        if lpips_value <= threshold:
                            move_to_transformed = True
                            break
                    
                    final_path = os.path.join(transformed_dir if move_to_transformed else discarded_dir, png_filename)
                    logging.info(f"Moving {png_filename} to {'transformed' if move_to_transformed else 'discarded'} directory.")
                    os.rename(output_png_path, final_path)
                    
                    results.append((png_filename, lpips_value))
                    image_counter += 1
    
    with open("lpips_results.csv", "w") as f:
        f.write("image_name,lpips_score\n")
        for img_name, score in results:
            f.write(f"{img_name},{score:.4f}\n")
    
    # print("Processing completed. Results saved in lpips_results.csv")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    # print(f"\033[33mExecution Time: {elapsed_time:.2f} seconds\033[0m")
    print(f"\033[33mProcessing completed. Execution Time: {elapsed_time:.2f} seconds\033[0m")

"""
el CSV está en una carpeta (manifest-1608669183333) que ya no contiene las carpetas DICOM, pero las rutas del CSV (File Location) están escritas 
como si sí lo hiciera (i.e., son relativas a esa carpeta). Sin embargo, tú has movido el CSV, y los archivos DICOM están en otra ruta totalmente distinta, 
externa y variable
"""


########
# Example usage
########

"""
process_dicom_folders(
    path_NBIA_Data='../../../../../ChestCT-NBIA/manifest-1608669183333',  # Path to the folder containing metadata.csv and Lung-PET-CT-Dx
    reference_images_paths=['Img_ref/Imagen_Ref1.png', 'Img_ref/Imagen_Ref2.png', 'Img_ref/Imagen_Ref3.png', 'Img_ref/Imagen_Ref4.png', 'Img_ref/Imagen_Ref5.png', 
                            'Img_ref/Imagen_Ref6.png', 'Img_ref/Imagen_Ref7.png', 'Img_ref/Imagen_Ref8.png', 'Img_ref/Imagen_Ref9.png', 'Img_ref/Imagen_Ref10.png'],
    discarded_reference_images_paths=['Img_ref/Imagen_Discarded_1.png', 'Img_ref/Imagen_Discarded_2.png', 'Img_ref/Imagen_Discarded_3.png', 'Img_ref/Imagen_Discarded_4.png', 'Img_ref/Imagen_Discarded_5.png'],
    transformed_dir='Data-Transformed/cancer',
    discarded_dir='Discarded/',
    threshold_lpips=0.3500,  
    threshold_psnr=15.0,  
    threshold_discard_lpips=0.2500,  
    threshold_discard_psnr=15.0)
"""
# Adjust the brightness of the images
def get_average_brightness(image_path):
    img = Image.open(image_path).convert('L')
    np_img = np.array(img)
    return np.mean(np_img)

def adjust_brightness(image_path, target_brightness=28):
    img = Image.open(image_path)
    current_brightness = get_average_brightness(image_path)
    if current_brightness < target_brightness:
        enhancer = ImageEnhance.Brightness(img)
        factor = target_brightness / current_brightness
        img = enhancer.enhance(factor)
        img.save(image_path)
        logging.info(f"{os.path.basename(image_path)}: Brightness adjusted from {current_brightness:.2f} to {target_brightness}")
    else:
        logging.info(f"{os.path.basename(image_path)}: Brightness is sufficient: {current_brightness:.2f}")


# Process all images in a directory and adjust their brightness
def adjust_brightness_in_directory(directory_path, target_brightness=28):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_path = os.path.join(root, file)
                try:
                    adjust_brightness(image_path, target_brightness)
                except Exception as e:
                    logging.warning(f"Error processing {image_path}: {e}")



