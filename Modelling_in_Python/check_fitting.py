
"""
Check overfitting vs underfitting using:
- Stratified cross-validation
- train vs CV metrics gap (return_train_score=True)
- optional learning curve plot

Works for: KNN, RandomForest, LogisticRegression
Optional: SMOTE / undersampling (only if you want it)

"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Optional imblearn (SMOTE / undersampling)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def load_and_prepare(csv_path: str, sep: str, encoding: str):
    """
    Expect a 'status' column with values including 'acquired' and 'closed'.
    Target: acquired=1, closed=0 (drops other statuses).
    Features: simple set + log funding if available.

    Adjust feature list to match your CSV columns if needed.
    """
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python")
    df.columns = df.columns.str.strip()

    if "status" not in df.columns:
        raise KeyError("Missing column: status")

    df["target"] = df["status"].map({"acquired": 1, "closed": 0})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Candidate features (edit freely)
    candidate = [
        "funding_total_usd",
        "venture",
        "funding_rounds",
        "angels",
        "seed",
        "round_A", "round_B", "round_C", "round_D",
        "founded_year",
    ]
    use_cols = [c for c in candidate if c in df.columns]

    if len(use_cols) == 0:
        raise KeyError("No usable feature columns found. Update `candidate` in load_and_prepare().")

    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=use_cols + ["target"]).copy()

    # log transform for funding if available
    if "funding_total_usd" in df.columns:
        df["log_funding"] = np.log1p(df["funding_total_usd"])
        feature_cols = ["log_funding"] + [c for c in use_cols if c != "funding_total_usd"]
    else:
        feature_cols = use_cols

    X = df[feature_cols]
    y = df["target"]

    print("Rows used for modelling:", len(df))
    print("Target distribution:\n", y.value_counts(normalize=False))
    print("Target distribution (%):\n", (y.value_counts(normalize=True) * 100).round(2))
    print("Features used:", feature_cols)

    return X, y


def build_model(model_name: str, args) -> object:
    """
    Returns a pipeline (with scaling where needed).
    If sampler != none and imblearn is available, uses ImbPipeline so sampling happens properly.
    """
    sampler = args.sampler

    # Choose pipeline class
    if sampler != "none":
        if not IMBLEARN_AVAILABLE:
            raise ImportError("Install imbalanced-learn to use SMOTE/undersampling: pip install imbalanced-learn")
        Pipe = ImbPipeline
    else:
        Pipe = SkPipeline

    steps = []

    # KNN & LogReg benefit from scaling; RF doesn't need it
    if model_name in ("knn", "logreg"):
        steps.append(("scaler", StandardScaler()))

    # Optional sampler
    if sampler == "smote":
        steps.append(("sampler", SMOTE(random_state=args.random_state)))
    elif sampler == "under":
        steps.append(("sampler", RandomUnderSampler(random_state=args.random_state)))

    # Model
    if model_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=args.k,
            weights=args.knn_weights,
            metric=args.knn_metric
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.depth if args.depth > 0 else None,
            class_weight=args.class_weight,
            random_state=args.random_state,
            n_jobs=-1
        )
    elif model_name == "logreg":
        model = LogisticRegression(
            max_iter=4000,
            class_weight=args.class_weight,
            random_state=args.random_state
        )
    else:
        raise ValueError("model must be one of: knn, rf, logreg")

    steps.append(("model", model))
    return Pipe(steps)


def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else (float(arr[0]), 0.0)


def diagnose(train_auc, cv_auc, train_f1, cv_f1):
    """
    Simple heuristics (rules of thumb):
    - Overfitting: train much better than CV (large positive gap)
    - Underfitting: both are low and close
    - OK: decent CV and moderate gap
    """
    gap_auc = train_auc - cv_auc
    gap_f1 = train_f1 - cv_f1

    # You can tweak these thresholds
    OVERFIT_GAP_AUC = 0.06
    OVERFIT_GAP_F1 = 0.08

    UNDERFIT_CV_AUC = 0.65
    UNDERFIT_CV_F1 = 0.55

    if (gap_auc >= OVERFIT_GAP_AUC) or (gap_f1 >= OVERFIT_GAP_F1):
        return "Likely OVERFITTING (train >> CV)"
    if (cv_auc <= UNDERFIT_CV_AUC and cv_f1 <= UNDERFIT_CV_F1 and gap_auc < 0.03 and gap_f1 < 0.05):
        return "Likely UNDERFITTING (train and CV both low)"
    return "Looks OK (no strong signs of over/underfitting)"


def run_cv_report(model, X, y, cv):
    # F1 for positive class = 1
    f1_pos = make_scorer(f1_score, pos_label=1)

    scoring = {
        "AUC": "roc_auc",
        "F1": f1_pos,
        "Precision": "precision",
        "Recall": "recall",
    }

    res = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    out = {}
    for m in scoring.keys():
        tr = res[f"train_{m}"]
        te = res[f"test_{m}"]
        out[m] = {
            "train_mean": float(np.mean(tr)),
            "train_std": float(np.std(tr, ddof=1)) if len(tr) > 1 else 0.0,
            "cv_mean": float(np.mean(te)),
            "cv_std": float(np.std(te, ddof=1)) if len(te) > 1 else 0.0,
        }
    return out


def maybe_plot_learning_curve(model, X, y, cv, out_path: str, scoring: str = "roc_auc"):
    import matplotlib.pyplot as plt

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="train")
    plt.plot(train_sizes, val_mean, label="validation")
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="Data/startups_new.csv")
    ap.add_argument("--sep", type=str, default=";")
    ap.add_argument("--encoding", type=str, default="latin1")

    ap.add_argument("--model", choices=["knn", "rf", "logreg"], required=True)

    # CV
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)

    # Sampling
    ap.add_argument("--sampler", choices=["none", "smote", "under"], default="none")

    # KNN params
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--knn_weights", choices=["uniform", "distance"], default="distance")
    ap.add_argument("--knn_metric", type=str, default="minkowski")

    # RF params
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--depth", type=int, default=20)  # <=0 means None
    ap.add_argument("--class_weight", type=str, default="balanced")  # "balanced" or None

    # Plot
    ap.add_argument("--plot_learning_curve", type=int, default=0)  # 1 = yes

    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    X, y = load_and_prepare(args.csv, args.sep, args.encoding)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    model = build_model(args.model, args)

    report = run_cv_report(model, X, y, cv)

    # Print nicely
    def fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"

    print("\n=== Cross-Validation Report (train vs CV) ===")
    for m in ["AUC", "F1", "Precision", "Recall"]:
        r = report[m]
        print(f"{m:9s}  train: {fmt(r['train_mean'], r['train_std'])}   CV: {fmt(r['cv_mean'], r['cv_std'])}   gap: {(r['train_mean']-r['cv_mean']):.3f}")

    diagnosis = diagnose(
        train_auc=report["AUC"]["train_mean"],
        cv_auc=report["AUC"]["cv_mean"],
        train_f1=report["F1"]["train_mean"],
        cv_f1=report["F1"]["cv_mean"],
    )
    print("\nDiagnosis:", diagnosis)

    if args.plot_learning_curve == 1:
        out_png = f"learning_curve_{args.model}.png"
        maybe_plot_learning_curve(model, X, y, cv, out_png, scoring="roc_auc")
        print(f"Saved learning curve plot to: {out_png}")


if __name__ == "__main__":
    main()
