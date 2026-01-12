import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# ---------------------------
# Automatically find dataset
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = [
    os.path.join(BASE_DIR, "..", "Data", "startups_new.csv"),
    os.path.join(BASE_DIR, "Data", "startups_new.csv"),
]

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

CSV_PATH = first_existing(CANDIDATES)
if CSV_PATH is None:
    raise FileNotFoundError("Could not find startup_new.csv in Data/")

print("Using dataset:", CSV_PATH)

# ---------------------------
# Helpers
# ---------------------------
def pct(x):
    return f"{x*100:.2f}%"

def rm_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": pct(accuracy_score(y_true, y_pred)),
        "pred_1_1": int(tp),
        "pred_1_0": int(fn),
        "pred_0_1": int(fp),
        "pred_0_0": int(tn),
        "class_recall_1_1": pct(recall_score(y_true, y_pred, pos_label=1)),
        "class_recall_0_0": pct(recall_score(y_true, y_pred, pos_label=0)),
        "class_precision_1_1": pct(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "class_precision_0_0": pct(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }

# ---------------------------
# Load & prepare data
# ---------------------------
df = pd.read_csv(CSV_PATH, sep=";", encoding="latin1", engine="python")
df.columns = df.columns.str.strip()

# Target: acquired=1, closed=0
df["target"] = df["status"].map({"acquired": 1, "closed": 0})
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

# Correlated features
features = [
    "funding_total_usd",
    "venture",
    "funding_rounds",
    "round_A", "round_B", "round_C", "round_D",
    "founded_year"
]
features = [c for c in features if c in df.columns]

for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=features).copy()

df["log_funding"] = np.log1p(df["funding_total_usd"])

X = df[["log_funding"] + [c for c in features if c != "funding_total_usd"]]
y = df["target"]

print("Rows used:", len(df))
print("Target distribution (1=acquired, 0=closed):")
print(y.value_counts(), "\n")

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# imbalance correction
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos

# ---------------------------
# Train XGBoost
# ---------------------------
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

m = rm_metrics(y_test, y_pred)

# ---------------------------
# Save RapidMiner-style CSV
# ---------------------------
out = pd.DataFrame([{
    "id": 1,
    **m,
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.1
}])

OUT_PATH = os.path.join(BASE_DIR, "xgb_results.csv")
out.to_csv(OUT_PATH, sep=";", index=False)

print("Saved:", OUT_PATH)
