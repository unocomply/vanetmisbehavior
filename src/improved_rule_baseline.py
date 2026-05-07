from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_features.csv"

BASE_FEATURES = [
    "spd_mag",
    "acl_mag",
    "pos_noise_diff",
    "spd_noise_diff",
    "acl_noise_diff",
    "hed_noise_diff",
    "combined_noise",
]


QUANTILES_TO_TEST = [0.85, 0.88, 0.90]


CUTOFF = 1

def evaluate(y_true, y_pred):
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\nAccuracy:", round(accuracy, 4))
    print("False Positive Rate:", round(fpr, 4))
    print("False Negative Rate:", round(fnr, 4))

    return accuracy, fpr, fnr, cm

def main():
    df = pd.read_csv(DATA_FILE)

    
    df["combined_noise"] = (
        df["pos_noise_diff"] +
        df["spd_noise_diff"] +
        df["acl_noise_diff"] +
        df["hed_noise_diff"]
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["attack"]
    )

    results = []

    for quantile in QUANTILES_TO_TEST:
        print("\n" + "=" * 60)
        print(f"Testing improved rule baseline: quantile={quantile}, cutoff={CUTOFF}")
        print("=" * 60)

        benign_train = train_df[train_df["attack"] == 0]

        thresholds = {
            feature: benign_train[feature].quantile(quantile)
            for feature in BASE_FEATURES
        }

        print("\nThresholds:")
        for feature, value in thresholds.items():
            print(f"{feature}: {value:.4f}")

        temp = val_df.copy()
        temp["rule_score"] = 0

        for feature in BASE_FEATURES:
            temp[f"{feature}_flag"] = (temp[feature] > thresholds[feature]).astype(int)
            temp["rule_score"] += temp[f"{feature}_flag"]

        temp["rule_pred"] = (temp["rule_score"] >= CUTOFF).astype(int)

        accuracy, fpr, fnr, cm = evaluate(temp["attack"], temp["rule_pred"])

        results.append({
            "quantile": quantile,
            "cutoff": CUTOFF,
            "accuracy": round(accuracy, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        })

    results_df = pd.DataFrame(results)

    print("\nImproved rule baseline summary:")
    print(results_df)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = results_dir / "improved_rule_baseline_summary.csv"
    results_df.to_csv(out_file, index=False)

    print("\nSaved improved results to:")
    print(out_file)

if __name__ == "__main__":
    main()