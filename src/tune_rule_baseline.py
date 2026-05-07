from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

QUANTILES = [0.90, 0.92, 0.95, 0.97, 0.99]
CUTOFFS = [1, 2, 3]

def compute_rule_score(df, thresholds):
    score = pd.Series(0, index=df.index)
    for feature in FEATURES:
        score += (df[feature] > thresholds[feature]).astype(int)
    return score

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2

    return {
        "accuracy": accuracy,
        "precision_attack": precision,
        "recall_attack": recall,
        "f1_attack": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "balanced_accuracy": balanced_accuracy,
    }

def main():
    df = pd.read_csv(DATA_FILE)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["attack"]
    )

    benign_train = train_df[train_df["attack"] == 0].copy()

    results = []

    for q in QUANTILES:
        thresholds = {
            feature: benign_train[feature].quantile(q)
            for feature in FEATURES
        }

        val_score = compute_rule_score(val_df, thresholds)

        for cutoff in CUTOFFS:
            val_pred = (val_score >= cutoff).astype(int)

            metrics = evaluate(val_df["attack"], val_pred)
            results.append({
                "quantile": q,
                "cutoff": cutoff,
                **metrics
            })

    results_df = pd.DataFrame(results)

    # sort by attack F1 first, then balanced accuracy
    results_df = results_df.sort_values(
        ["f1_attack", "balanced_accuracy", "recall_attack"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    print("\nTop tuning results:")
    print(results_df.head(10))

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = results_dir / "rule_tuning_results.csv"
    results_df.to_csv(out_file, index=False)

    print(f"\nSaved tuning results to: {out_file}")

if __name__ == "__main__":
    main()