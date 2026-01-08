import argparse
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Samplers
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# ------------------------------------------------------------
# DATA LOADING & PREPARATION
# ------------------------------------------------------------
def load_and_prepare(csv_path: str, sep: str, encoding: str):
    print("Loading data...")
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python")

    df.columns = df.columns.str.strip()

    required = ["funding_total_usd", "funding_rounds", "founded_at", "Status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df[required].copy()
    df = df.dropna()

    df["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")
    df["company_age"] = 2025 - df["founded_at"].dt.year

    for c in ["funding_total_usd", "funding_rounds", "Status"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["log_funding"] = np.log1p(df["funding_total_usd"])
    df = df.dropna()

    df = df[df["Status"].isin([0, 1])]
    df["Status"] = df["Status"].astype(int)

    X = df[["log_funding", "funding_rounds", "company_age"]]
    y = df["Status"]

    print(f"Loaded {df.shape[0]} rows")
    print("Class distribution:")
    print(y.value_counts(normalize=True))
    print()

    return X, y


# ------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------
def apply_sampler(X_train_scaled, y_train, sampler, random_state):
    if sampler == "none":
        return X_train_scaled, y_train

    if not IMBLEARN_AVAILABLE:
        raise ImportError("Install imbalanced-learn to use sampling.")

    if sampler == "smote":
        print("Applying SMOTE...")
        sm = SMOTE(random_state=random_state)
        return sm.fit_resample(X_train_scaled, y_train)

    if sampler == "under":
        print("Applying undersampling...")
        ru = RandomUnderSampler(random_state=random_state)
        return ru.fit_resample(X_train_scaled, y_train)

    raise ValueError("Invalid sampler")


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", auc)

    return auc


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="startups.csv")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--sampler", choices=["none", "smote", "under"], default="none")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    start = time.time()

    # Load data
    X, y = load_and_prepare(args.csv, args.sep, args.encoding)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sampling (only for Logistic & KNN)
    X_train_final, y_train_final = apply_sampler(
        X_train_scaled, y_train, args.sampler, args.random_state
    )

    # --------------------------------------------------------
    # 1) LOGISTIC REGRESSION
    # --------------------------------------------------------
    print("\nTraining Logistic Regression...")
    log_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if args.sampler == "none" else None,
        random_state=args.random_state
    )
    log_model.fit(X_train_final, y_train_final)
    log_auc = evaluate_model(
        "Logistic Regression",
        log_model,
        X_test_scaled,
        y_test,
        args.threshold
    )

    # --------------------------------------------------------
    # 2) RANDOM FOREST (NO SCALING)
    # --------------------------------------------------------
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_auc = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test,
        args.threshold
    )

    # --------------------------------------------------------
    # 3) KNN
    # --------------------------------------------------------
    print("\nTraining KNN...")
    knn_model = KNeighborsClassifier(
        n_neighbors=25,
        weights="distance"
    )
    knn_model.fit(X_train_final, y_train_final)
    knn_auc = evaluate_model(
        "KNN",
        knn_model,
        X_test_scaled,
        y_test,
        args.threshold
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n==============================")
    print("MODEL COMPARISON (ROC AUC)")
    print("==============================")
    print(f"Logistic Regression: {log_auc:.3f}")
    print(f"Random Forest:       {rf_auc:.3f}")
    print(f"KNN:                 {knn_auc:.3f}")
    print("==============================")
    print(f"Sampler: {args.sampler} | Threshold: {args.threshold}")
    print(f"Runtime: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
