import matplotlib.pyplot as plt
import numpy as np

"""
PENDIENTE CON DATOS DE training_log_wgan.csv c
"""

# Simulación de datos (reemplazar con tus datos)
epochs = np.arange(0, 12)  # Reemplaza con el número de épocas
loss_d = [-0.278, -0.125, -0.501, -0.781, -1.059, -0.457, -0.566, -0.627, -0.490, -0.673, -1.310, -0.980]
loss_g = [0.013, 0.034, 0.284, 0.468, 0.627, 0.493, 0.467, 0.470, 0.444, 0.515, 0.698, 0.572]

# Visualización de pérdidas
def plot_losses(epochs, loss_d, loss_g):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_d, label="Discriminator Loss (Loss_D)", color="red")
    plt.plot(epochs, loss_g, label="Generator Loss (Loss_G)", color="blue")
    plt.title("Loss Evolution of WGAN", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# Función para visualizar imágenes generadas
def plot_generated_images(generator, epoch, latent_dim=100, n=4):
    """
    Visualiza imágenes generadas por el generador.
    """
    noise = np.random.normal(0, 1, (n * n, latent_dim))  # Ruido aleatorio
    generated_images = generator.predict(noise)  # Imágenes generadas

    # Normaliza imágenes
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(generated_images[i], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Simulación de un generador (reemplaza con tu modelo)
class DummyGenerator:
    def predict(self, noise):
        return np.random.rand(noise.shape[0], 64, 64)  # Simula imágenes 64x64

# Llamada a funciones
plot_losses(epochs, loss_d, loss_g)

# Visualización de imágenes generadas (reemplazar DummyGenerator con tu generador)
generator = DummyGenerator()
plot_generated_images(generator, epoch=1000, latent_dim=100, n=4)
