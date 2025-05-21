import os
import time
import pydicom
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from datetime import datetime
import cv2  

"""
This script generate a reduced dataset of DICOM images from the NBIA database.
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

def calculate_psnr(img1, img2):
    """Calculates PSNR between two images"""
    return cv2.PSNR(img1, img2)

def process_dicom_folders(path_NBIA_Data, transformed_dir, discarded_dir, discarded_reference_images_paths, discard_percentage=0.15, psnr_threshold=30):
    ensure_directory_exists(transformed_dir)
    ensure_directory_exists(discarded_dir)

    results = []
    image_counter = 1

    metadata_path = os.path.join(path_NBIA_Data, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    metadata = metadata[metadata['Study UID'].isin(valid_study_descriptions)]

    start_time = time.time()

    for _, row in metadata.iterrows():
        folder_path = os.path.join(path_NBIA_Data, row['File Location'])
        if not os.path.exists(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            dicom_files = sorted([f for f in files if f.lower().endswith('.dcm')])

            # Calculate the number of files to discard
            num_to_discard = int(len(dicom_files) * discard_percentage)
            dicom_files = dicom_files[num_to_discard:-num_to_discard]  # Exclude 15% from the start and end

            for file in dicom_files:
                dicom_path = os.path.join(root, file)
                png_filename = f"ChestCT_{image_counter}.png"
                output_png_path = os.path.join(transformed_dir, png_filename)

                dicom_to_png(dicom_path, output_png_path)

                final_path = os.path.join(transformed_dir, png_filename)

                # Load the transformed image
                transformed_img = cv2.imread(final_path)

                # Compare with reference images and calculate PSNR
                discard_image = False
                for reference_img_path in discarded_reference_images_paths:
                    reference_img = cv2.imread(reference_img_path)

                    psnr_value = calculate_psnr(transformed_img, reference_img)

                    # If PSNR is higher than the threshold, move to 'discarded' folder
                    if psnr_value >= psnr_threshold:
                        # Move the image to discarded
                        os.rename(output_png_path, os.path.join(discarded_dir, png_filename))
                        discard_image = True
                        break

                # If not discarded, move to 'transformed'
                if not discard_image:
                    os.rename(output_png_path, final_path)
                    results.append((png_filename))

                image_counter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing completed. Execution Time: {elapsed_time:.2f} seconds")

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
        img = enhancer.enhance(factor)  # Increase the brightness of the image

        img.save(image_path)  # Overwrite the original image with adjusted brightness
        print(f"Brightness adjusted from {current_brightness:.2f} to {target_brightness}")
    else:
        print(f"The brightness of the image is sufficient: {current_brightness:.2f}")


transformed_dir = 'Data-Transformed/cancer'


for filename in os.listdir(transformed_dir):
    if filename.lower().endswith('.png'):  
        image_path = os.path.join(transformed_dir, filename)
        adjust_brightness(image_path, target_brightness=25)


process_dicom_folders(
    path_NBIA_Data='../../../../../ChestCT-NBIA/manifest-1608669183333', 
    transformed_dir=transformed_dir,  
    discarded_dir='Data-Discarded',  
    discarded_reference_images_paths=[
        'Img_ref/Imagen_Discarded_1.png', 
        'Img_ref/Imagen_Discarded_2.png', 
        'Img_ref/Imagen_Discarded_3.png', 
        'Img_ref/Imagen_Discarded_4.png', 
        'Img_ref/Imagen_Discarded_5.png'
    ],
    discard_percentage=0.15,  
    psnr_threshold=15.0 
)
