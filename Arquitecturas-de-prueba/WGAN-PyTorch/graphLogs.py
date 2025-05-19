import matplotlib.pyplot as plt
import pandas as pd

"""
This script has some hardcoded paths. It has been used only for testing purposes.
"""
file_path = "training_log_wgan.csv"  # Yoy may want to change this to a relative path or use a config file
df = pd.read_csv(file_path)

# Convert columns to numeric, forcing errors to NaN
df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
df["Loss_D"] = pd.to_numeric(df["Loss_D"], errors="coerce")
df["Loss_G"] = pd.to_numeric(df["Loss_G"], errors="coerce")

# Filter out rows with NaN values
df = df.dropna(subset=["Epoch", "Loss_D", "Loss_G"])

plt.figure(figsize=(10, 6))

# Graph the losses
plt.plot(df["Epoch"], df["Loss_D"], label="Loss Discriminator", marker='o', linestyle='--', color="red")
plt.plot(df["Epoch"], df["Loss_G"], label="Loss Generator", marker='s', linestyle='-', color="blue")

plt.title("Evolución de las pérdidas del Discriminador y Generador", fontsize=16, fontweight='bold')
plt.xlabel("Época", fontsize=14)
plt.ylabel("Pérdida", fontsize=14)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

