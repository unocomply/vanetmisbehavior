import pandas as pd

df = pd.read_csv("results/final_rule_baseline_predictions.csv")

print("Sample predictions:\n")
print(df[["attack", "rule_score", "rule_pred", "attack_type"]].head(20))

print("\nSummary:")
print("Total samples:", len(df))
print("Predicted attacks:", df["rule_pred"].sum())
print("Actual attacks:", df["attack"].sum())

import matplotlib.pyplot as plt

df["rule_score"].hist(bins=10)
plt.title("Rule Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()