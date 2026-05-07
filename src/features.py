from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_dev.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_features.csv"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pos_mag"] = np.sqrt(df["pos_0"]**2 + df["pos_1"]**2)
    df["spd_mag"] = np.sqrt(df["spd_0"]**2 + df["spd_1"]**2)
    df["acl_mag"] = np.sqrt(df["acl_0"]**2 + df["acl_1"]**2)
    df["hed_mag"] = np.sqrt(df["hed_0"]**2 + df["hed_1"]**2)

    df["pos_noise_diff"] = np.sqrt(
        (df["pos_noise_0"] - df["pos_0"])**2 +
        (df["pos_noise_1"] - df["pos_1"])**2
    )
    df["spd_noise_diff"] = np.sqrt(
        (df["spd_noise_0"] - df["spd_0"])**2 +
        (df["spd_noise_1"] - df["spd_1"])**2
    )
    df["acl_noise_diff"] = np.sqrt(
        (df["acl_noise_0"] - df["acl_0"])**2 +
        (df["acl_noise_1"] - df["acl_1"])**2
    )
    df["hed_noise_diff"] = np.sqrt(
        (df["hed_noise_0"] - df["hed_0"])**2 +
        (df["hed_noise_1"] - df["hed_1"])**2
    )

    return df

def main():
    df = pd.read_csv(IN_FILE)
    df = add_features(df)
    df.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE)
    print(df.head())

if __name__ == "__main__":
    main()