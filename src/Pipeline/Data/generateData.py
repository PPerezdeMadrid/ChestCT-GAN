import os
import time
import pydicom
import torch
import lpips
import logging
from PIL import Image, ImageEnhance
from torchvision import transforms
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2


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

def calculate_ssim(image1, image2):
    # Convert images to grayscale if they are in color
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    score, _ = ssim(image1_gray, image2_gray, full=True)
    return score

def setup_logging():
    log_filename = f"data_generated_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def process_dicom_folders(path_NBIA_Data, reference_images_paths, discarded_reference_images_paths, transformed_dir, discarded_dir, threshold_lpips=0.3500, threshold_ssim=0.7, threshold_discard_lpips=0.3500, threshold_discard_ssim=0.7):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = lpips.LPIPS(net='vgg').to(device)
    
    ensure_directory_exists(transformed_dir)
    ensure_directory_exists(discarded_dir)

    results = []
    image_counter = 1  # Counter for naming images as ChestCT_X.png

    # Read the metadata CSV
    metadata_path = os.path.join(path_NBIA_Data, 'metadata.csv')  # Path to the CSV file
    metadata = pd.read_csv(metadata_path)
    
    # Filter rows with valid Study Description
    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    metadata = metadata[metadata['Study UID'].isin(valid_study_descriptions)]
    
    logging.info(f"Found {len(metadata)} records with valid Study UID.")
    start_time = time.time()

    # Iterate over metadata rows with a progress bar
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing folders"):
        # Get the folder path from Download Timestamp
        folder_path = os.path.join(path_NBIA_Data, row['File Location'])
        
        if not os.path.exists(folder_path):
            logging.info(f"Folder not found: {folder_path}")
            continue
        
        logging.info(f"Processing folder: {folder_path}")
        
        # Walk through DICOM images in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    png_filename = f"ChestCT_{image_counter}.png"  # Using image_counter to name the files
                    output_png_path = os.path.join(transformed_dir, png_filename)  # Save in transformed_dir
                    
                    # Convert the DICOM image to PNG
                    dicom_to_png(dicom_path, output_png_path)
                    
                    # Initialize flags to check if the image should be moved
                    move_to_transformed = False
                    move_to_discarded = False
                    
                    # Compare the DICOM image with each reference image (for transformation)
                    for reference_image_path in reference_images_paths:
                        # Calculate LPIPS
                        lpips_value = calculate_lpips(model, reference_image_path, output_png_path, device)
                        logging.info(f"LPIPS score for {png_filename} with {reference_image_path}: {lpips_value:.4f}")
                        
                        # Load images for SSIM calculation
                        ref_image = cv2.imread(reference_image_path)
                        test_image = cv2.imread(output_png_path)

                        # Calculate SSIM
                        ssim_value = calculate_ssim(ref_image, test_image)
                        logging.info(f"SSIM score for {png_filename} with {reference_image_path}: {ssim_value:.4f}")
                        
                        # Check if both LPIPS and SSIM meet the thresholds (for transformed images)
                        if lpips_value <= threshold_lpips and ssim_value >= threshold_ssim:
                            move_to_transformed = True
                            break  # If both LPIPS and SSIM are good, move to transformed directory
                    
                    # Compare the DICOM image with each reference image (for discarded images)
                    for discarded_reference_image_path in discarded_reference_images_paths:
                        # Calculate LPIPS for discarded references
                        lpips_value_discarded = calculate_lpips(model, discarded_reference_image_path, output_png_path, device)
                        logging.info(f"LPIPS score for {png_filename} with discarded reference {discarded_reference_image_path}: {lpips_value_discarded:.4f}")
                        
                        # Load images for SSIM calculation (for discarded references)
                        discarded_ref_image = cv2.imread(discarded_reference_image_path)
                        test_image = cv2.imread(output_png_path)

                        # Calculate SSIM for discarded references
                        ssim_value_discarded = calculate_ssim(discarded_ref_image, test_image)
                        logging.info(f"SSIM score for {png_filename} with discarded reference {discarded_reference_image_path}: {ssim_value_discarded:.4f}")
                        
                        # Check if both LPIPS and SSIM meet the thresholds (for discarded images)
                        if lpips_value_discarded <= threshold_discard_lpips and ssim_value_discarded >= threshold_discard_ssim:
                            move_to_discarded = True
                            break  # If both LPIPS and SSIM meet the thresholds for discarded, move to discarded directory
                    
                    # Move the image to the appropriate folder based on both LPIPS and SSIM
                    if move_to_discarded:
                        final_path = os.path.join(discarded_dir, png_filename)
                        logging.info(f"Moving {png_filename} to discarded directory.")
                    elif move_to_transformed:
                        final_path = os.path.join(transformed_dir, png_filename)
                        logging.info(f"Moving {png_filename} to transformed directory.")
                    else:
                        final_path = os.path.join(discarded_dir, png_filename)
                        logging.info(f"Moving {png_filename} to discarded directory.")
                    
                    os.rename(output_png_path, final_path)
                    
                    # Store the result
                    results.append((png_filename, lpips_value, ssim_value, lpips_value_discarded, ssim_value_discarded))
                    image_counter += 1  # Increment the image counter
    
    # Save results to a CSV file
    with open("lpips_ssim_results.csv", "w") as f:
        f.write("image_name,lpips_score,ssim_score,discarded_lpips_score,discarded_ssim_score\n")
        for img_name, lpips_score, ssim_score, discarded_lpips_score, discarded_ssim_score in results:
            f.write(f"{img_name},{lpips_score:.4f},{ssim_score:.4f},{discarded_lpips_score:.4f},{discarded_ssim_score:.4f}\n")
    
    logging.info("Processing completed. Results saved to lpips_ssim_results.csv")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    logging.info(f"\033[33mExecution Time: {elapsed_time:.2f} seconds\033[0m")
    print(f"\033[33mProcessing completed. Execution Time: {elapsed_time:.2f} seconds\033[0m")


process_dicom_folders(
    path_NBIA_Data='../../../../../ChestCT-NBIA/manifest-1608669183333',  # Path to the folder containing metadata.csv and Lung-PET-CT-Dx
    reference_images_paths=['Imagen_Ref1.png', 'Imagen_Ref2.png', 'Imagen_Ref3.png', 'Imagen_Ref4.png', 'Imagen_Ref5.png', 'Imagen_Ref6.png', 'Imagen_Ref7.png', 'Imagen_Ref8.png'],
    discarded_reference_images_paths=['Imagen_Discarded_1.png', 'Imagen_Discarded_2.png', 'Imagen_Discarded_3.png', 'Imagen_Discarded_4.png'],
    transformed_dir='Data-Transformed/cancer',
    discarded_dir='Discarded/',
    threshold_lpips=0.3500,  
    threshold_ssim=0.3,  
    threshold_discard_lpips=0.4000,  
    threshold_discard_ssim=0.25)

# Ajustar el brillo de las imágenes
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
        img = enhancer.enhance(factor)  # Aumentar el brillo de la imagen

        img.save(image_path)  # Sobrescribe la imagen original con el brillo ajustado
        print(f"Brillo ajustado de {current_brightness:.2f} a {target_brightness}")
    else:
        print(f"El brillo de la imagen es suficiente: {current_brightness:.2f}")

# Directorio de imágenes transformadas
transformed_dir = 'Data-Transformed/cancer'

# Iterar sobre las imágenes en el directorio
for filename in os.listdir(transformed_dir):
    if filename.lower().endswith('.png'):  # Solo trabajar con archivos PNG
        image_path = os.path.join(transformed_dir, filename)
        adjust_brightness(image_path, target_brightness=25)