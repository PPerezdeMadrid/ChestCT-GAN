import matplotlib.pyplot as plt
import pandas as pd

"""
This script has some hardcoded paths. It has been used only for testing purposes.
This part is included as an Annex to the final degree project
"""
file_path = "evaluation/evaluation_dcgan_64/training_log_dcgan_2025-03-17.csv"
df = pd.read_csv(file_path)


df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
df["Loss_D"] = pd.to_numeric(df["Loss_D"], errors="coerce")
df["Loss_G"] = pd.to_numeric(df["Loss_G"], errors="coerce")

df = df.dropna(subset=["Epoch", "Loss_D", "Loss_G", "D(G(z))_Fake"])

plt.figure(figsize=(10, 6))

plt.plot(df["Epoch"], df["Loss_D"], label="Loss Discriminator", marker='o', linestyle='--', color="red")
plt.plot(df["Epoch"], df["Loss_G"], label="Loss Generator", marker='s', linestyle='-', color="blue")

plt.title("Evolución de las pérdidas del Discriminador y Generador", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Pérdida", fontsize=14)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

