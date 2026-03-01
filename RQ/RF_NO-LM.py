import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance


def get_gcs_class(category: str) -> int:
    if category == "gcs_2":
        return 2
    if category == "gcs_3":
        return 3
    if category == "gcs_45_clean":
        return 5
    return 0


def text_feats(s: str) -> pd.Series:
    if not isinstance(s, str):
        s = ""
    s = s.strip()
    if not s:
        return pd.Series(
            {
                "num_tokens": 0,
                "num_chars": 0,
                "mean_token_length": 0.0,
                "unique_ratio": 0.0,
                "repetition_ratio": 0.0,
                "alpha_ratio": 0.0,
                "digit_ratio": 0.0,
                "punct_ratio": 0.0,
            }
        )

    tokens = re.findall(r"\b\w+\b", s.lower(), flags=re.UNICODE)
    num_tokens = len(tokens)
    num_chars = len(s)

    if num_tokens == 0:
        mean_len = 0.0
        unique_ratio = 0.0
        repetition_ratio = 0.0
        alpha_ratio = 0.0
        digit_ratio = 0.0
    else:
        mean_len = float(np.mean([len(t) for t in tokens]))
        unique_ratio = float(len(set(tokens)) / num_tokens)
        repetition_ratio = float(1.0 - unique_ratio)
        alpha_ratio = float(sum(t.isalpha() for t in tokens) / num_tokens)
        digit_ratio = float(sum(any(ch.isdigit() for ch in t) for t in tokens) / num_tokens)

    punct_count = sum(ch in ".,;:!?\"'()-[]{}" for ch in s)
    punct_ratio = punct_count / max(1, num_chars)

    return pd.Series(
        {
            "num_tokens": num_tokens,
            "num_chars": num_chars,
            "mean_token_length": mean_len,
            "unique_ratio": unique_ratio,
            "repetition_ratio": repetition_ratio,
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
            "punct_ratio": punct_ratio,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to features_validation_set.csv (defaults to ../results/features_validation_set.csv relative to this file)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(project_root, "results")

    validation_csv_path = args.csv or os.path.join(results_dir, "features_validation_set.csv")

    print("Loading:", validation_csv_path)
    df = pd.read_csv(validation_csv_path)

    df["gcs_class"] = df["category"].apply(get_gcs_class)
    df = df[df["gcs_class"] != 0].copy()

    print("\nClass counts:")
    print(df["gcs_class"].value_counts())

    if df["gcs_class"].nunique() < 2:
        raise ValueError("Need at least two classes to train")

    if "transcription" not in df.columns:
        raise KeyError("Missing 'transcription' column.")

    df = df.reset_index(drop=True)
    df = pd.concat([df, df["transcription"].apply(text_feats)], axis=1)

    whisper_silero_features = ["silero_speech_prob", "whisper_no_speech_prob", "avg_logprob"]
    text_features_1 = [
        "num_tokens",
        "num_chars",
        "mean_token_length",
        "unique_ratio",
        "repetition_ratio",
        "alpha_ratio",
        "digit_ratio",
        "punct_ratio",
    ]
    feature_cols = whisper_silero_features + text_features_1

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    for c in feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X = df[feature_cols].copy()
    y = df["gcs_class"].copy()

    print("\nX shape:", X.shape, "y shape:", y.shape)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain counts:\n", y_train.value_counts())
    print("\nHoldout counts:\n", y_holdout.value_counts())

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
    )

    print("\nTraining RF")
    rf.fit(X_train, y_train)

    pred = rf.predict(X_holdout)

    labels = [2, 3, 5]
    names = ["GCS 2", "GCS 3", "GCS 4/5"]

    print("\nHoldout report:")
    print(classification_report(y_holdout, pred, labels=labels, target_names=names, zero_division=0))

    acc = accuracy_score(y_holdout, pred)
    bal = balanced_accuracy_score(y_holdout, pred)
    print(f"Accuracy: {acc:.2%}")
    print(f"Balanced Accuracy: {bal:.2%}")

    cm = confusion_matrix(y_holdout, pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nPermutation importance (holdout, balanced accuracy):")
    perm = permutation_importance(
        rf,
        X_holdout,
        y_holdout,
        n_repeats=20,
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
    print(imp)

    plt.figure(figsize=(9, 4))
    imp.head(12).plot(kind="bar")
    plt.ylabel("Importance")
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.show()

    print("\n5-fold CV (balanced accuracy):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    print(f"mean={scores.mean():.3f}, std={scores.std():.3f}")


if __name__ == "__main__":
    main()