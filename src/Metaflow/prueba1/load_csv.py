import pandas as pd

def load_csv(csv_file):
    """Carga los datos del CSV."""
    df = pd.read_csv(csv_file)
    print("Datos cargados:\n", df.head())
    return df
