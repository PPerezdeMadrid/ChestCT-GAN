import matplotlib.pyplot as plt
import pandas as pd
import json

# Cargar el archivo CSV
file_name = "training_log_dcgan_2025-03-23.csv"  
with open("config.json", "r") as file:
    config = json.load(file)
    file_path = config["model"]["evaluation_dcgan"] + "/" + file_name

df = pd.read_csv(file_path)

# Convertir columnas relevantes a tipos numéricos
df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
df["Loss_D"] = pd.to_numeric(df["Loss_D"], errors="coerce")
df["Loss_G"] = pd.to_numeric(df["Loss_G"], errors="coerce")

# Filtrar filas válidas (en caso de que haya errores en los datos)
df = df.dropna(subset=["Epoch", "Loss_D", "Loss_G"])

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Graficar Loss_D y Loss_G
plt.plot(df["Epoch"], df["Loss_D"], label="Loss Discriminator", marker='o', linestyle='--', color="red")
plt.plot(df["Epoch"], df["Loss_G"], label="Loss Generator", marker='s', linestyle='-', color="blue")

# Configurar títulos y etiquetas
plt.title("Evolución de las pérdidas del Discriminador y Generador", fontsize=16, fontweight='bold')
plt.xlabel("Época", fontsize=14)
plt.ylabel("Pérdida", fontsize=14)

# Mejoras visuales
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

