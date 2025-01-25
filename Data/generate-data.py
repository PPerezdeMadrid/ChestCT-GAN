import os, time
import pydicom
import torch
import lpips
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

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

def process_dicom_folders(path_NBIA_Data, reference_image_path, transformed_dir, discarded_dir, threshold=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net='vgg').to(device)
    
    ensure_directory_exists(transformed_dir)
    ensure_directory_exists(discarded_dir)
    
    results = []

    # Read the metadata CSV
    metadata_path = os.path.join(path_NBIA_Data, 'metadata.csv')  # Path to the CSV file
    metadata = pd.read_csv(metadata_path)
    
    # Filter rows with valid Study Description
    valid_study_descriptions = ["Chest", "ThoraxAThoraxRoutine Adult", "Chest 3D IMR", "Chest 3D"]
    metadata = metadata[metadata['Study UID'].isin(valid_study_descriptions)]
    
    print(f"Found {len(metadata)} records with valid Study UID.")
    start_time = time.time()
    
    for _, row in metadata.iterrows():
        # Get the folder path from Download Timestamp
        folder_path = os.path.join(path_NBIA_Data, row['File Location'])
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
        
        print(f"Processing folder: {folder_path}")
        
        # Walk through DICOM images in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    png_filename = f"{os.path.splitext(file)[0]}.png"
                    output_png_path = os.path.join(transformed_dir, png_filename)
                    
                    # Convert the DICOM image to PNG
                    dicom_to_png(dicom_path, output_png_path)
                    
                    # Calculate LPIPS value
                    lpips_value = calculate_lpips(model, reference_image_path, output_png_path, device)
                    print(f"LPIPS score for {png_filename}: {lpips_value:.4f}")
                    
                    # Move the image to the appropriate folder based on LPIPS
                    if lpips_value < threshold:
                        final_path = os.path.join(transformed_dir, png_filename)
                        print(f"Moving {png_filename} to transformed directory.")
                    else:
                        final_path = os.path.join(discarded_dir, png_filename)
                        print(f"Moving {png_filename} to discarded directory.")
                    
                    os.rename(output_png_path, final_path)
                    
                    # Store the result
                    results.append((png_filename, lpips_value))
    
    # Save results to a CSV file
    with open("lpips_results.csv", "w") as f:
        f.write("image_name,lpips_score\n")
        for img_name, score in results:
            f.write(f"{img_name},{score:.4f}\n")
    
    print("Processing completed. Results saved to lpips_results.csv")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Execution Time: {elapsed_time:.2f} seconds") 


# Example usage
process_dicom_folders(
    path_NBIA_Data='../../../../ChestCT-NBIA/manifest-1608669183333',  # Path to the folder containing metadata.csv and Lung-PET-CT-Dx
    reference_image_path='Ejemplo_de_imagen.png',
    transformed_dir='Data-Transformed/',
    discarded_dir='Discarded/',
    threshold=0.360
)