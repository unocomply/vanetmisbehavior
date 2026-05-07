from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_features.csv"

FEATURES = [
    "spd_mag",
    "acl_mag",
    "pos_noise_diff",
    "spd_noise_diff",
    "acl_noise_diff",
    "hed_noise_diff",
]

FINAL_QUANTILE = 0.90
FINAL_CUTOFF = 1

def main():
    df = pd.read_csv(DATA_FILE)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["attack"]
    )

    benign_train = train_df[train_df["attack"] == 0].copy()

    thresholds = {
        feature: benign_train[feature].quantile(FINAL_QUANTILE)
        for feature in FEATURES
    }

    print("Final thresholds:")
    for feature, value in thresholds.items():
        print(f"{feature}: {value:.4f}")

    val_df = val_df.copy()
    val_df["rule_score"] = 0

    for feature in FEATURES:
        val_df[f"{feature}_flag"] = (val_df[feature] > thresholds[feature]).astype(int)
        val_df["rule_score"] += val_df[f"{feature}_flag"]

    val_df["rule_pred"] = (val_df["rule_score"] >= FINAL_CUTOFF).astype(int)

    y_true = val_df["attack"]
    y_pred = val_df["rule_pred"]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\nAccuracy:", round(accuracy, 4))
    print("False Positive Rate:", round(fpr, 4))
    print("False Negative Rate:", round(fnr, 4))

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = results_dir / "final_rule_baseline_predictions.csv"
    summary_file = results_dir / "final_rule_baseline_metrics.txt"

    cols_to_save = [
        "attack",
        "attack_type",
        "rule_score",
        "rule_pred",
    ] + [f"{feature}_flag" for feature in FEATURES]

    val_df[cols_to_save].to_csv(out_file, index=False)

    with open(summary_file, "w") as f:
        f.write(f"Final rule baseline\n")
        f.write(f"quantile={FINAL_QUANTILE}\n")
        f.write(f"cutoff={FINAL_CUTOFF}\n")
        f.write(f"accuracy={accuracy:.4f}\n")
        f.write(f"fpr={fpr:.4f}\n")
        f.write(f"fnr={fnr:.4f}\n")
        f.write(f"confusion_matrix=\n{cm}\n")

    print("\nSaved files:")
    print(out_file)
    print(summary_file)

if __name__ == "__main__":
    main()