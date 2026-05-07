from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "veremi" / "veremi.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_dev.csv"

def main():
    df = pd.read_csv(RAW_FILE)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # smaller balanced working set
    dev = (
        df.groupby("attack", group_keys=False)
          .apply(lambda x: x.sample(n=100000, random_state=42))
          .reset_index(drop=True)
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dev.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)
    print("Shape:", dev.shape)
    print(dev["attack"].value_counts())

if __name__ == "__main__":
    main()