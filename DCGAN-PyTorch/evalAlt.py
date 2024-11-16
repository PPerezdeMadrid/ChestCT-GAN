
"""
#############################################  
        Inception Score (IS)
 ############################################   
"""

def get_inception_score(images, splits=10):
    """Calcula el Inception Score para un conjunto de imágenes."""
    scores = []
    N = len(images)
    
    for i in range(splits):
        part = images[i * (N // splits): (i + 1) * (N // splits)]
        with torch.no_grad():
            pred = inception_model(part)  # Pasar las imágenes por Inception
            pred = F.softmax(pred, dim=1)  # Aplicar softmax

        scores.append(pred.cpu().numpy())
    
    scores = np.concatenate(scores, axis=0)
    # Calcular el Inception Score
    kl_divergence = scores * (np.log(scores) - np.log(np.mean(scores, axis=0)))
    is_score = np.exp(np.mean(np.sum(kl_divergence, axis=1)))
    
    return is_score

"""
# Generar imágenes
num_images = 100  # Número de imágenes a generar
fake_images = generate_images(netG, num_images, params['nz'])

# Preprocesar imágenes
preprocessed_images = preprocess_images(fake_images)

# Cargar Inception-v3
# inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
inception_model.eval()

# Calcular el Inception Score
is_score = get_inception_score(preprocessed_images)
# Print pero en bonito :)
print(f"{'-' * 30}")
print(f"{'Inception Score':^30}")
print(f"{'-' * 30}")
print(f"{'Score:':<20} {is_score:.4f}") 
print(f"{'-' * 30}")
"""

"""
#############################################  
    Fréchet Inception Distance (FID)
 ############################################   
"""

def get_activations(images, model, batch_size=64, dims=2048):
    n_batches = images.shape[0] // batch_size
    pred_arr = np.empty((n_batches, dims))
    for i in range(n_batches):
        batch = images[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            pred = model(batch)
        pred_arr[i] = pred.numpy()
    return pred_arr

def calculate_fid(real_activations, fake_activations):
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_value

"""
# Calcular FID
def calculate_fid_score(real_images, fake_images):
    # Preprocesar imágenes
    real_images = preprocess_images(real_images)
    fake_images = preprocess_images(fake_images)

    # Obtener activaciones
    real_activations = get_activations(real_images, inception_model)
    fake_activations = get_activations(fake_images, inception_model)

    # Calcular FID
    fid_score = calculate_fid(real_activations, fake_activations)
    return fid_score


# Cargar las imágenes reales
real_images, _ = next(iter(dataloader))  # Obtener un batch de imágenes reales

# Calcular FID
fake_images = fake_images.numpy()  # Convertir el tensor a un array de NumPy
fid_score = calculate_fid_score(real_images, fake_images)

# Imprimir el resultado
print(f"{'-' * 30}")
print(f"{'Fréchet Inception Distance':^30}")  # Centrar el título
print(f"{'-' * 30}")
print(f"{'FID Score:':<20} {fid_score:.4f}")  # Alinear la etiqueta y mostrar el score
print(f"{'-' * 30}")
"""
