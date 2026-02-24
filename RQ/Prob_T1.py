import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier


project_root = ".."
results_dir = os.path.join(project_root, "results")
csv_path = os.path.join(results_dir, "features_validation_set.csv")  #change if needed

print("Loading:", csv_path)
df = pd.read_csv(csv_path)

def get_gcs_class(category: str) -> int:
    if category == "gcs_2": return 2
    if category == "gcs_3": return 3
    if category == "gcs_45_clean": return 5
    return 0

df["gcs_class"] = df["category"].apply(get_gcs_class)
df = df[df["gcs_class"] != 0].copy()

print("\nClass counts:\n", df["gcs_class"].value_counts())

if df["gcs_class"].nunique() < 2:
    raise ValueError("Only one class ispresent")


feature_cols = ["avg_logprob", "lexical_validity", "perplexity"]
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing cols: {missing}")


df["perplexity"] = df["perplexity"].fillna(df["perplexity"].median())
p99 = df["perplexity"].quantile(0.99)
df["log_perplexity"] = np.log1p(df["perplexity"].clip(upper=p99))

#replace perplexity with log_perplexity
feature_cols = ["avg_logprob", "lexical_validity", "log_perplexity"]

for c in feature_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

X = df[feature_cols]
y = df["gcs_class"]


X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


gb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=410,
    random_state=42
)

gb.fit(X_train, y_train)
print("Training done")


pred = gb.predict(X_holdout)

labels = [2, 3, 5]
names = ["GCS 2", "GCS 3", "GCS 4/5"]

print("\nClassification report (on holdout):")
print(classification_report(y_holdout, pred, labels=labels, target_names=names, zero_division=0))

print("Accuracy:", accuracy_score(y_holdout, pred))
print("Balanced accuracy:", balanced_accuracy_score(y_holdout, pred))

cm = confusion_matrix(y_holdout, pred, labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Holdout)")
plt.tight_layout()
plt.show()


from sklearn.inspection import permutation_importance

perm = permutation_importance(gb, X_holdout, y_holdout, n_repeats=20, random_state=42)
imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("\nPermutation importances:\n", imp)

plt.figure(figsize=(6,4))
imp.plot(kind="bar")
plt.ylabel("Mean importance")
plt.title("Permutation Importance ")
plt.tight_layout()
plt.show()


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb, X, y, cv=cv, scoring="balanced_accuracy")
print(f"\n5-fold CV balanced accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")