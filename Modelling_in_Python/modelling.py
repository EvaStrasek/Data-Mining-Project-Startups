import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# imblearn (SMOTE / undersampling)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# -------------------------
# Helpers (RapidMiner-style)
# -------------------------
def pct(x: float) -> str:
    return f"{x * 100:.2f}%"

def rapidminer_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": pct(accuracy_score(y_true, y_pred)),
        "pred_1_1": int(tp),  # TP
        "pred_1_0": int(fn),  # FN
        "pred_0_1": int(fp),  # FP
        "pred_0_0": int(tn),  # TN
        "class_recall_1_1": pct(recall_score(y_true, y_pred, pos_label=1)),
        "class_recall_0_0": pct(recall_score(y_true, y_pred, pos_label=0)),
        "class_precision_1_1": pct(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "class_precision_0_0": pct(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }

def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_bool_list_rm(s: str):
    # expects e.g. "FALSE,TRUE"
    return [x.strip().upper() == "TRUE" for x in s.split(",") if x.strip()]


# -------------------------
# Data loading / prep
# -------------------------
def load_and_prepare(csv_path: str, sep: str, encoding: str, year_now: int):
    """
    Crunchbase-style input (semicolon separated usually):
      - 'status' column contains: acquired / closed / operating (we keep only acquired+closed)
      - numeric features include: funding_total_usd, venture, funding_rounds, round_A..round_D, founded_year
    Output:
      X: dataframe with selected features
      y: target (1=acquired, 0=closed)
    """
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python")
    df.columns = df.columns.str.strip()

    # Target mapping
    if "status" not in df.columns:
        raise KeyError("Missing column: status")

    df["target"] = df["status"].map({"acquired": 1, "closed": 0})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Candidate features (based on your correlation results)
    candidate = [
        "funding_total_usd",
        "venture",
        "funding_rounds",
        "round_A", "round_B", "round_C", "round_D",
        "founded_year",
    ]

    must_have = ["funding_total_usd", "venture", "funding_rounds", "founded_year"]
    missing_must = [c for c in must_have if c not in df.columns]
    if missing_must:
        raise KeyError(f"Missing required columns for modelling: {missing_must}")

    use_cols = [c for c in candidate if c in df.columns]

    # Convert to numeric
    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing rows
    df = df.dropna(subset=use_cols + ["target"]).copy()

    # log transform for funding
    df["log_funding"] = np.log1p(df["funding_total_usd"])

    feature_cols = ["log_funding"] + [c for c in use_cols if c != "funding_total_usd"]

    X = df[feature_cols]
    y = df["target"]

    print("Rows used for modelling:", len(df))
    print("Target distribution:\n", y.value_counts())

    return X, y


# -------------------------
# KNN CSV (exact column order)
# -------------------------
def build_knn_df(X_train, y_train, X_test, y_test, k_list, sampler, random_state):
    rows = []
    cols = [
        "id","accuracy","pred_1_1","pred_1_0","pred_0_1","pred_0_0",
        "class_recall_1_1","class_recall_0_0","class_precision_1_1","class_precision_0_0","k"
    ]

    for idx, k in enumerate(k_list, start=1):
        if sampler != "none":
            if not IMBLEARN_AVAILABLE:
                raise ImportError("pip install imbalanced-learn (required for SMOTE/undersampling)")

            steps = [("scaler", StandardScaler())]
            if sampler == "smote":
                steps.append(("sampler", SMOTE(random_state=random_state)))
            elif sampler == "under":
                steps.append(("sampler", RandomUnderSampler(random_state=random_state)))

            steps.append(("model", KNeighborsClassifier(n_neighbors=k, weights="distance")))
            model = Pipeline(steps)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
            model = KNeighborsClassifier(n_neighbors=k, weights="distance")
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

        m = rapidminer_metrics(y_test, y_pred)
        rows.append({
            "id": idx,
            "accuracy": m["accuracy"],
            "pred_1_1": m["pred_1_1"],
            "pred_1_0": m["pred_1_0"],
            "pred_0_1": m["pred_0_1"],
            "pred_0_0": m["pred_0_0"],
            "class_recall_1_1": m["class_recall_1_1"],
            "class_recall_0_0": m["class_recall_0_0"],
            "class_precision_1_1": m["class_precision_1_1"],
            "class_precision_0_0": m["class_precision_0_0"],
            "k": int(k),
        })

    return pd.DataFrame(rows)[cols]


# -------------------------
# RF CSV (exact column order) with RapidMiner-default prepruning
# -------------------------
def build_rf_df(
    X_train, y_train, X_test, y_test,
    trees_list, depth_list, prepruning_list,
    random_state
):
    """
    RapidMiner prepruning defaults (commonly used / documented):
      - minimal leaf size = 2
      - minimal size for split = 4

    Here we replicate the part that is directly comparable in sklearn:
      - min_samples_leaf = 2
      - min_samples_split = 4
    """
    RM_MIN_SAMPLES_LEAF = 2
    RM_MIN_SAMPLES_SPLIT = 4

    rows = []
    cols = [
        "id","accuracy","pred_1_1","pred_1_0","pred_0_1","pred_0_0",
        "class_recall_1_1","class_recall_0_0","class_precision_1_1","class_precision_0_0",
        "number_of_trees","max_depth","prepruning"
    ]

    idx = 1
    for depth in depth_list:
        for trees in trees_list:
            for prepruning in prepruning_list:
                if prepruning:
                    min_leaf = RM_MIN_SAMPLES_LEAF
                    min_split = RM_MIN_SAMPLES_SPLIT
                else:
                    min_leaf = 1
                    min_split = 2

                rf = RandomForestClassifier(
                    n_estimators=int(trees),
                    max_depth=int(depth),
                    min_samples_leaf=int(min_leaf),
                    min_samples_split=int(min_split),
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                m = rapidminer_metrics(y_test, y_pred)
                rows.append({
                    "id": idx,
                    "accuracy": m["accuracy"],
                    "pred_1_1": m["pred_1_1"],
                    "pred_1_0": m["pred_1_0"],
                    "pred_0_1": m["pred_0_1"],
                    "pred_0_0": m["pred_0_0"],
                    "class_recall_1_1": m["class_recall_1_1"],
                    "class_recall_0_0": m["class_recall_0_0"],
                    "class_precision_1_1": m["class_precision_1_1"],
                    "class_precision_0_0": m["class_precision_0_0"],
                    "number_of_trees": int(trees),
                    "max_depth": int(depth),
                    "prepruning": "TRUE" if prepruning else "FALSE",
                })
                idx += 1

    return pd.DataFrame(rows)[cols]


# -------------------------
# Logistic Regression CSV (NO C column; exact order requested)
# -------------------------
def build_logreg_df(X_train, y_train, X_test, y_test, sampler, random_state):
    rows = []
    cols = [
        "id","accuracy","pred_1_1","pred_1_0","pred_0_1","pred_0_0",
        "class_recall_1_1","class_recall_0_0","class_precision_1_1","class_precision_0_0"
    ]

    idx = 1

    if sampler != "none":
        if not IMBLEARN_AVAILABLE:
            raise ImportError("pip install imbalanced-learn (required for SMOTE/undersampling)")

        steps = [("scaler", StandardScaler())]
        if sampler == "smote":
            steps.append(("sampler", SMOTE(random_state=random_state)))
        elif sampler == "under":
            steps.append(("sampler", RandomUnderSampler(random_state=random_state)))

        # keep balanced weighting even when using sampler (consistency)
        steps.append(("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        )))
        model = Pipeline(steps)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        )
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

    m = rapidminer_metrics(y_test, y_pred)
    rows.append({
        "id": idx,
        "accuracy": m["accuracy"],
        "pred_1_1": m["pred_1_1"],
        "pred_1_0": m["pred_1_0"],
        "pred_0_1": m["pred_0_1"],
        "pred_0_0": m["pred_0_0"],
        "class_recall_1_1": m["class_recall_1_1"],
        "class_recall_0_0": m["class_recall_0_0"],
        "class_precision_1_1": m["class_precision_1_1"],
        "class_precision_0_0": m["class_precision_0_0"],
    })

    return pd.DataFrame(rows)[cols]


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Export RapidMiner-style CSV results for KNN, RF, and Logistic Regression.")

    # Defaults adjusted for your Crunchbase CSVs
    parser.add_argument("--csv", type=str, default="../Data/startups_new.csv")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--encoding", type=str, default="latin1")
    parser.add_argument("--year_now", type=int, default=2025)  # not used anymore, kept for compatibility

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)

    # Sampling for KNN + Logistic
    parser.add_argument("--sampler", choices=["none", "smote", "under"], default="none")

    # KNN params
    parser.add_argument("--k_list", type=str, default="5,10,15,20,25")

    # RF params
    parser.add_argument("--trees_list", type=str, default="50,100,200,300,400,500")
    parser.add_argument("--depth_list", type=str, default="10,15,20")
    parser.add_argument("--prepruning_list", type=str, default="FALSE,TRUE")

    # Output paths
    parser.add_argument("--out_knn", type=str, default="knn_results.csv")
    parser.add_argument("--out_rf", type=str, default="rf_results.csv")
    parser.add_argument("--out_logreg", type=str, default="logreg_results.csv")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv} (working dir: {os.getcwd()})")

    X, y = load_and_prepare(args.csv, args.sep, args.encoding, args.year_now)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    k_list = parse_int_list(args.k_list)
    trees_list = parse_int_list(args.trees_list)
    depth_list = parse_int_list(args.depth_list)
    prepruning_list = parse_bool_list_rm(args.prepruning_list)

    knn_df = build_knn_df(X_train, y_train, X_test, y_test, k_list, args.sampler, args.random_state)
    rf_df = build_rf_df(
        X_train, y_train, X_test, y_test,
        trees_list, depth_list, prepruning_list,
        args.random_state
    )
    log_df = build_logreg_df(X_train, y_train, X_test, y_test, args.sampler, args.random_state)

    knn_df.to_csv(args.out_knn, index=False, sep=";")
    rf_df.to_csv(args.out_rf, index=False, sep=";")
    log_df.to_csv(args.out_logreg, index=False, sep=";")

    print("Saved CSV files:")
    print(" -", args.out_knn)
    print(" -", args.out_rf)
    print(" -", args.out_logreg)


if __name__ == "__main__":
    main()
