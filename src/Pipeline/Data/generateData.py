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
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
PSNR tenga prioridad sobre LPIPS, y que solo si el PSNR supera su umbral, se evalúe LPIPS
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
def process_dicom_folders(path_NBIA_Data, reference_images_paths, discarded_reference_images_paths, transformed_dir, discarded_dir, threshold_lpips=0.3500, threshold_psnr=20.0, threshold_discard_lpips=0.3500, threshold_discard_psnr=20.0):
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
            for file in files:
                if file.lower().endswith('.dcm'):
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
                            print(f"PSNR (transf) {png_filename} vs {reference_image_path}: {psnr_value:.4f}")
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
                            print(f"PSNR (discard) {png_filename} vs {discarded_reference_image_path}: {psnr_value_discarded:.4f}")
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
    logging.info(f"\033[33mExecution Time: {elapsed_time:.2f} seconds\033[0m")
    print(f"\033[33mProcessing completed. Execution Time: {elapsed_time:.2f} seconds\033[0m")
"""

def process_dicom_folders(path_NBIA_Data, reference_images_paths, discarded_reference_images_paths, transformed_dir, discarded_dir, threshold_lpips=0.3500, threshold_psnr=20.0, threshold_discard_lpips=0.3500, threshold_discard_psnr=20.0):
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
                        print(f"PSNR (transf) {png_filename} vs {reference_image_path}: {psnr_value:.4f}")
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
                        print(f"PSNR (discard) {png_filename} vs {discarded_reference_image_path}: {psnr_value_discarded:.4f}")
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
    logging.info(f"\033[33mExecution Time: {elapsed_time:.2f} seconds\033[0m")
    print(f"\033[33mProcessing completed. Execution Time: {elapsed_time:.2f} seconds\033[0m")


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
