import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Visualización de las pérdidas del Discriminador y Generador.")
    parser.add_argument('--log_file', type=str, required=True, help="Archivo CSV con los logs de entrenamiento.")
    return parser.parse_args()


def load_data(log_file):
    with open("config.json", "r") as file:
        config = json.load(file)
        file_path = config["model"]["evaluation_dcgan"] + "/" + log_file

    df = pd.read_csv(file_path)

    df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
    df["Loss_D"] = pd.to_numeric(df["Loss_D"], errors="coerce")
    df["Loss_G"] = pd.to_numeric(df["Loss_G"], errors="coerce")

    df = df.dropna(subset=["Epoch", "Loss_D", "Loss_G"])

    return df

def plot_losses(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df["Epoch"], df["Loss_D"], label="Loss Discriminador", marker='o', linestyle='--', color="red")
    plt.plot(df["Epoch"], df["Loss_G"], label="Loss Generador", marker='s', linestyle='-', color="blue")

    plt.title("Evolución de las pérdidas del Discriminador y Generador", fontsize=16, fontweight='bold')
    plt.xlabel("Época", fontsize=14)
    plt.ylabel("Pérdida", fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    df = load_data(args.log_file)
    plot_losses(df)

if __name__ == "__main__":
    main()
