from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "veremi_features.csv"

FEATURES = [
    "type",
    "pos_mag",
    "spd_mag",
    "acl_mag",
    "hed_mag",
    "pos_noise_diff",
    "spd_noise_diff",
    "acl_noise_diff",
    "hed_noise_diff",
]

def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df["attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, preds))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\nFalse Positive Rate:", round(fpr, 4))
    print("False Negative Rate:", round(fnr, 4))

    importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature importances:")
    print(importance_df)

if __name__ == "__main__":
    main()