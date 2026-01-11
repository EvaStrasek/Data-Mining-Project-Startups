import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def engineer_features(df: pd.DataFrame, top_markets: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    # Clean column names
    df.columns = df.columns.str.strip()
    # Fix the known typo: frim_age -> firm_age
    if "frim_age" in df.columns and "firm_age" not in df.columns:
        df = df.rename(columns={"frim_age": "firm_age"})

    required = [
        "funding_total_usd", "market", "funding_rounds",
        "seed", "venture", "equity_crowdfunding", "undisclosed", "debt_financing",
        "angel", "grant", "private_equity", "post_ipo_equity", "post_ipo_debt",
        "secondary_market", "product_crowdfunding",
        "round_A", "round_B", "round_C", "round_D", "round_E", "round_F", "round_G", "round_H",
        "firm_age", "Status"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}\nAvailable: {df.columns.tolist()}")

    # Keep only needed columns
    df = df[required].copy()

    # Ensure numeric for numeric columns (everything except market)
    for c in df.columns:
        if c != "market":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    # Status must be 0/1
    df = df[df["Status"].isin([0, 1])].copy()
    df["Status"] = df["Status"].astype(int)

    # --- Feature engineering ---
    df["log_funding"] = np.log1p(df["funding_total_usd"])

    # Aggregate funding types into interpretable groups
    df["vc_funding"] = df[["venture", "private_equity"]].sum(axis=1)
    df["early_funding"] = df[["seed", "angel", "grant"]].sum(axis=1)
    df["crowdfunding"] = df[["equity_crowdfunding", "product_crowdfunding"]].sum(axis=1)
    df["debt_funding"] = df[["debt_financing", "post_ipo_debt"]].sum(axis=1)
    df["other_funding"] = df[["undisclosed", "secondary_market", "post_ipo_equity"]].sum(axis=1)

    # Aggregate rounds
    df["early_rounds"] = df[["round_A", "round_B"]].sum(axis=1)
    df["late_rounds"] = df[["round_C", "round_D", "round_E", "round_F", "round_G", "round_H"]].sum(axis=1)

    # Market encoding: top N + other
    top = df["market"].value_counts().nlargest(top_markets).index
    df["market_clean"] = df["market"].where(df["market"].isin(top), "other")
    market_dummies = pd.get_dummies(df["market_clean"], prefix="market")
    df = pd.concat([df, market_dummies], axis=1)

    # Final feature set (compact + interpretable)
    feature_cols = [
        "log_funding",
        "funding_rounds",
        "firm_age",
        "vc_funding",
        "early_funding",
        "crowdfunding",
        "debt_funding",
        "other_funding",
        "early_rounds",
        "late_rounds",
    ] + list(market_dummies.columns)

    X = df[feature_cols]
    y = df["Status"]

    return X, y


def evaluate(name: str, model, X_test, y_test, threshold: float) -> float:
    # predict_proba needed for ROC AUC + thresholding
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="startups_python.csv")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_markets", type=int, default=8)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, sep=args.sep, encoding=args.encoding, engine="python")

    X, y = engineer_features(df, top_markets=args.top_markets)

    print("\nClass distribution (overall):")
    print(y.value_counts(normalize=True).rename("proportion"))

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Scaling for Logistic + KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # 1) Logistic Regression
    # ----------------------------
    log_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",  # helps with imbalance
        random_state=args.random_state,
    )
    print("\nTraining Logistic Regression...")
    log_model.fit(X_train_scaled, y_train)
    log_auc = evaluate("Logistic Regression", log_model, X_test_scaled, y_test, args.threshold)

    # ----------------------------
    # 2) Random Forest
    # ----------------------------
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1
    )
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)  # unscaled
    rf_auc = evaluate("Random Forest", rf_model, X_test, y_test, args.threshold)

    # ----------------------------
    # 3) KNN
    # ----------------------------
    knn_model = KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        metric="minkowski"
    )
    print("\nTraining KNN...")
    knn_model.fit(X_train_scaled, y_train)
    knn_auc = evaluate("KNN", knn_model, X_test_scaled, y_test, args.threshold)

    # Summary
    print("\n==============================")
    print("Model comparison (ROC AUC)")
    print("==============================")
    print(f"Logistic Regression: {log_auc:.3f}")
    print(f"Random Forest:       {rf_auc:.3f}")
    print(f"KNN:                 {knn_auc:.3f}")
    print(f"Threshold used:      {args.threshold}")
    print("==============================\n")


if __name__ == "__main__":
    main()
