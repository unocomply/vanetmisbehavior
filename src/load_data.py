from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "veremi" / "veremi.csv"

def load_veremi():
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

if __name__ == "__main__":
    df = load_veremi()

    print("Loaded successfully.")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nAttack counts:")
    print(df["attack"].value_counts(dropna=False))

    print("\nAttack type counts:")
    print(df["attack_type"].value_counts(dropna=False).head(10))