import os
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ ML imports (for CV comparison + ROC) ------------------------
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

import matplotlib.pyplot as plt


# ------------------------ Robust base dir ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


# ------------------------ File paths ------------------------
CANDIDATE_KNN_RM = [
    os.path.join(BASE_DIR, "Data", "knn_results_13012026.csv"),
    os.path.join(BASE_DIR, "..", "Data", "knn_results_13012026.csv"),
]
CANDIDATE_KNN_PY = [
    os.path.join(BASE_DIR, "Data", "knn_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "knn_results.csv"),
]

CANDIDATE_RF_RM = [
    os.path.join(BASE_DIR, "Data", "results_random_forest_13012026.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_random_forest_13012026.csv"),
]
CANDIDATE_RF_PY = [
    os.path.join(BASE_DIR, "Data", "rf_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "rf_results.csv"),
]

CANDIDATE_LR_RM = [
    os.path.join(BASE_DIR, "Data", "results_logreg.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_logreg.csv"),
    os.path.join(BASE_DIR, "Data", "results_logistic_regression.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_logistic_regression.csv"),
]
CANDIDATE_LR_PY = [
    os.path.join(BASE_DIR, "Data", "logreg_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "logreg_results.csv"),
    os.path.join(BASE_DIR, "Data", "results_logreg_python.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_logreg_python.csv"),
]

# XGBoost results candidates (CSV results page)
CANDIDATE_XGB = [
    os.path.join(BASE_DIR, "Data", "xgb_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "xgb_results.csv"),
    os.path.join(BASE_DIR, "xgb_results.csv"),
]

CANDIDATE_SAMPLE = [
    os.path.join(BASE_DIR, "Data", "startups_new.csv"),
    os.path.join(BASE_DIR, "..", "Data", "startups_new.csv"),
]

# NEW: dataset for CV evaluation (acquired vs closed)
CANDIDATE_CV_DATASET = [
    os.path.join(BASE_DIR, "Data", "startups_new.csv"),
    os.path.join(BASE_DIR, "..", "Data", "startups_new.csv"),
    os.path.join(BASE_DIR, "Data", "startups_python.csv"),      # fallback
    os.path.join(BASE_DIR, "..", "Data", "startups_python.csv"),
]

KNN_RM_PATH = first_existing(CANDIDATE_KNN_RM)
KNN_PY_PATH = first_existing(CANDIDATE_KNN_PY)

RF_RM_PATH = first_existing(CANDIDATE_RF_RM)
RF_PY_PATH = first_existing(CANDIDATE_RF_PY)

LR_RM_PATH = first_existing(CANDIDATE_LR_RM)
LR_PY_PATH = first_existing(CANDIDATE_LR_PY)

XGB_PATH = first_existing(CANDIDATE_XGB)

SAMPLE_PATH = first_existing(CANDIDATE_SAMPLE)
CV_DATASET_PATH = first_existing(CANDIDATE_CV_DATASET)


# ------------------------ Helpers ------------------------
def pct_to_float(series):
    s = series.astype(str).str.strip()
    is_pct = s.str.endswith("%", na=False)
    nums = pd.to_numeric(s.where(~is_pct, s.str.replace("%", "", regex=False)), errors="coerce")
    nums = np.where(is_pct, nums / 100, nums)
    return pd.Series(nums, index=series.index)


def load_results(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    # Normalize pruning/prepruning naming
    if "pruning" in df.columns and "prepruning" not in df.columns:
        df = df.rename(columns={"pruning": "prepruning"})

    # Normalize True/False values
    if "prepruning" in df.columns:
        df["prepruning"] = df["prepruning"].astype(str).str.strip().str.upper()

    # Percent columns -> floats
    for col in [
        "accuracy",
        "class_recall_1_1",
        "class_recall_0_0",
        "class_precision_1_1",
        "class_precision_0_0",
    ]:
        if col in df.columns:
            df[col] = pct_to_float(df[col])

    # Confusion-matrix counts -> ints
    for col in ["pred_1_1", "pred_1_0", "pred_0_1", "pred_0_0"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def confusion_table_from_row(row: pd.Series):
    """
    Uses your CSV convention:
      pred_1_1 = TP, pred_1_0 = FN, pred_0_1 = FP, pred_0_0 = TN
    """
    TP = int(row["pred_1_1"])
    FN = int(row["pred_1_0"])
    FP = int(row["pred_0_1"])
    TN = int(row["pred_0_0"])

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0

    precision_1 = TP / (TP + FP) if (TP + FP) else 0.0
    precision_0 = TN / (TN + FN) if (TN + FN) else 0.0
    recall_1 = TP / (TP + FN) if (TP + FN) else 0.0
    recall_0 = TN / (TN + FP) if (TN + FP) else 0.0

    table = pd.DataFrame(
        {
            "true 0": [TN, FP, f"{recall_0*100:.2f}%"],
            "true 1": [FN, TP, f"{recall_1*100:.2f}%"],
            "class precision": [f"{precision_0*100:.2f}%", f"{precision_1*100:.2f}%", ""],
        },
        index=["pred. 0", "pred. 1", "class recall"],
    )
    return acc, table


def best_knn_k(df: pd.DataFrame) -> int:
    best_row = df.loc[df["accuracy"].astype(float).idxmax()]
    return int(best_row["k"])


def best_rf_config(df: pd.DataFrame) -> tuple[int, int, str]:
    best_row = df.loc[df["accuracy"].astype(float).idxmax()]
    trees = int(best_row["number_of_trees"])
    depth = int(best_row["max_depth"])
    prepruning = str(best_row["prepruning"]).strip().upper()
    return trees, depth, prepruning


def load_sample_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin1")
    df.columns = df.columns.str.strip()

    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.strip()

    numeric_cols = [c for c in df.columns if c != "market"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    if "funding_total_usd" in df.columns:
        df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    for c in ["Status", "firm_age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def format_money(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def show_dual_results(title_top: str, row_top: pd.Series | None, title_bottom: str, row_bottom: pd.Series | None):
    if row_top is not None:
        st.subheader(title_top)
        acc_top, table_top = confusion_table_from_row(row_top)
        st.write(f"accuracy: **{acc_top*100:.2f}%**")
        st.dataframe(table_top, use_container_width=True)
        with st.expander("Row details (top)"):
            st.write(row_top)

    if row_bottom is not None:
        st.subheader(title_bottom)
        acc_bot, table_bot = confusion_table_from_row(row_bottom)
        st.write(f"accuracy: **{acc_bot*100:.2f}%**")
        st.dataframe(table_bot, use_container_width=True)
        with st.expander("Row details (bottom)"):
            st.write(row_bottom)

    if row_top is None and row_bottom is None:
        st.warning("No matching result row found for this selection.")


# ------------------------ CV helpers ------------------------
@st.cache_data(show_spinner=False)
def load_cv_dataset(path: str, sep=";", encoding="latin1"):
    df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
    df.columns = df.columns.str.strip()

    if "status" not in df.columns:
        raise KeyError("Dataset for CV must contain column: status")

    df["target"] = df["status"].map({"acquired": 1, "closed": 0})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

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
    if not use_cols:
        raise KeyError("No usable feature columns found for CV dataset. Update candidate list.")

    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=use_cols + ["target"]).copy()

    if "funding_total_usd" in df.columns:
        df["log_funding"] = np.log1p(df["funding_total_usd"])
        feature_cols = ["log_funding"] + [c for c in use_cols if c != "funding_total_usd"]
    else:
        feature_cols = use_cols

    X = df[feature_cols].values
    y = df["target"].values
    return df, X, y, feature_cols


def build_cv_models(
    random_state: int,
    knn_k: int,
    rf_depth: int,
    rf_trees: int,
    dt_depth: int,
    gbt_lr: float,
    gbt_estimators: int,
    xgb_lr: float,
    xgb_estimators: int,
    xgb_depth: int,
):
    models = {}

    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    models["KNN (uniform)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=knn_k,
            weights="uniform"
        ))
    ])

    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=dt_depth if dt_depth > 0 else None,
        class_weight="balanced",
        random_state=random_state
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=rf_depth if rf_depth > 0 else None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["Gradient Boosted Trees (sklearn)"] = GradientBoostingClassifier(
        learning_rate=gbt_lr,
        n_estimators=gbt_estimators,
        random_state=random_state
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_estimators,
            learning_rate=xgb_lr,
            max_depth=xgb_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    return models


def cv_metrics_table(models: dict, X, y, cv):
    scoring = {"AUC": "roc_auc", "F1": "f1", "Precision": "precision", "Recall": "recall"}
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"Model": name}
        for m in scoring.keys():
            vals = res[f"test_{m}"]
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def cv_roc_curve(model, X, y, cv):
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    aucv = roc_auc_score(y, y_proba)
    return fpr, tpr, float(aucv)


# ------------------------ Streamlit UI ------------------------
st.set_page_config(layout="wide")
st.title("Startups prediction and modelling")

# Load CSV-results datasets (if they exist)
knn_rm_df = load_results(KNN_RM_PATH) if KNN_RM_PATH else None
knn_py_df = load_results(KNN_PY_PATH) if KNN_PY_PATH else None

rf_rm_df = load_results(RF_RM_PATH) if RF_RM_PATH else None
rf_py_df = load_results(RF_PY_PATH) if RF_PY_PATH else None

lr_rm_df = load_results(LR_RM_PATH) if LR_RM_PATH else None
lr_py_df = load_results(LR_PY_PATH) if LR_PY_PATH else None

xgb_df = load_results(XGB_PATH) if XGB_PATH else None

sample_df = load_sample_dataset(SAMPLE_PATH) if SAMPLE_PATH else None


# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Choose view",
    ["Sample data", "Model Comparison", "kNN", "Random Forest", "Logistic Regression", "XGBoost"],
    key="view_choice",
)
st.sidebar.divider()

# Reset button (for selections)
if st.sidebar.button("Reset to best accuracy"):
    df_for_knn = knn_rm_df if knn_rm_df is not None else knn_py_df
    if df_for_knn is not None and "k" in df_for_knn.columns and "accuracy" in df_for_knn.columns:
        st.session_state["knn_k"] = best_knn_k(df_for_knn)

    df_for_rf = rf_rm_df if rf_rm_df is not None else rf_py_df
    if df_for_rf is not None and all(c in df_for_rf.columns for c in ["number_of_trees", "max_depth", "prepruning", "accuracy"]):
        bt, bd, bp = best_rf_config(df_for_rf)
        st.session_state["rf_trees"] = bt
        st.session_state["rf_depth"] = bd
        st.session_state["rf_prepruning"] = bp

    st.rerun()


# ------------------------ CV Comparison view ------------------------
if view == "Model Comparison":
    st.header("Model Comparison")

    if CV_DATASET_PATH is None:
        st.error("Could not find dataset for cross-validation (startups_new.csv / startups_python.csv).")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_CV_DATASET])
        st.stop()

    st.sidebar.subheader("Cross-validation settings")
    folds = st.sidebar.slider("Folds", 3, 10, 5, 1, key="cv_folds")
    rs = st.sidebar.number_input("Random state", value=42, step=1, key="cv_rs")

    st.sidebar.subheader("Model hyperparameters (CV)")
    knn_k = st.sidebar.slider("KNN k", 3, 50, 20, 1, key="cv_knn_k")
    rf_depth = st.sidebar.slider("RF max_depth (0=None)", 0, 30, 6, 1, key="cv_rf_depth")
    rf_trees = st.sidebar.slider("RF n_estimators", 50, 600, 300, 50, key="cv_rf_trees")
    dt_depth = st.sidebar.slider("Decision Tree max_depth (0=None)", 0, 30, 6, 1, key="cv_dt_depth")

    gbt_lr = st.sidebar.slider("GBT learning_rate", 0.01, 0.5, 0.1, 0.01, key="cv_gbt_lr")
    gbt_estimators = st.sidebar.slider("GBT n_estimators", 50, 600, 200, 50, key="cv_gbt_estimators")

    xgb_lr = st.sidebar.slider("XGB learning_rate", 0.01, 0.5, 0.1, 0.01, key="cv_xgb_lr")
    xgb_estimators = st.sidebar.slider("XGB n_estimators", 50, 800, 300, 50, key="cv_xgb_estimators")
    xgb_depth = st.sidebar.slider("XGB max_depth", 2, 10, 4, 1, key="cv_xgb_depth")

    show_rocs = st.sidebar.checkbox("Show ROC curve(s)", value=True, key="cv_show_rocs")
    roc_mode = st.sidebar.radio("ROC mode", ["Selected model", "All models"], index=0, key="cv_roc_mode")

    try:
        df_cv, X, y, feature_cols = load_cv_dataset(CV_DATASET_PATH)
    except Exception as e:
        st.error(f"Failed to load CV dataset: {e}")
        st.stop()

    st.caption(f"Loaded CV dataset from: {CV_DATASET_PATH}")
    st.write(f"**Rows used:** {len(df_cv)}")
    vc = pd.Series(y).value_counts()
    target_table = pd.DataFrame({
        "Class": ["Acquired (1)", "Closed (0)"],
        "Count": ["2419", "1439"],
        "Share (%)": ["62.7%", "37.3%"]
    })

    st.markdown("**Target distribution:**")
    st.table(target_table)
    st.markdown(
    "**Features used:** "
    "`total_funding`, `venture`, `funding_rounds`, `seed`, `angels`, "
    "`round_A`, `round_B`, `round_C`, `round_D`, `founded_year`")

    if not XGBOOST_AVAILABLE:
        st.info("XGBoost is not available (missing `xgboost`). Other models will still be evaluated.")

    cv = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(rs))

    models = build_cv_models(
        random_state=int(rs),
        knn_k=int(knn_k),
        rf_depth=int(rf_depth),
        rf_trees=int(rf_trees),
        dt_depth=int(dt_depth),
        gbt_lr=float(gbt_lr),
        gbt_estimators=int(gbt_estimators),
        xgb_lr=float(xgb_lr),
        xgb_estimators=int(xgb_estimators),
        xgb_depth=int(xgb_depth),
    )

    with st.spinner("Running cross-validation for all models..."):
        res_df = cv_metrics_table(models, X, y, cv)

    pretty = pd.DataFrame({
        "Model": res_df["Model"],
        "AUC": res_df.apply(lambda r: f"{r['AUC_mean']:.3f} ± {r['AUC_std']:.3f}", axis=1),
        "F1": res_df.apply(lambda r: f"{r['F1_mean']:.3f} ± {r['F1_std']:.3f}", axis=1),
        "Precision": res_df.apply(lambda r: f"{r['Precision_mean']:.3f} ± {r['Precision_std']:.3f}", axis=1),
        "Recall": res_df.apply(lambda r: f"{r['Recall_mean']:.3f} ± {r['Recall_std']:.3f}", axis=1),
    })

    # Sort by AUC mean (not the formatted string)
    res_df_sorted = res_df.sort_values(by="AUC_mean", ascending=False)
    pretty = pretty.set_index("Model").loc[res_df_sorted["Model"]].reset_index()

    st.subheader("Final comparison (CV)")
    st.dataframe(pretty, use_container_width=True)

    if show_rocs:
        st.subheader("ROC curve(s) (cross-validated probabilities)")
        model_names = list(models.keys())
        selected = st.selectbox("Select model for ROC", model_names, index=0, key="cv_selected_model")

        if roc_mode == "Selected model":
            m = models[selected]
            with st.spinner(f"Computing ROC for: {selected} ..."):
                fpr, tpr, aucv = cv_roc_curve(m, X, y, cv)

            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"{selected} (AUC={aucv:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            st.pyplot(fig, clear_figure=True)

        else:
            fig = plt.figure()
            with st.spinner("Computing ROC for all models..."):
                for name, m in models.items():
                    try:
                        fpr, tpr, aucv = cv_roc_curve(m, X, y, cv)
                        plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")
                    except Exception:
                        pass
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
            st.pyplot(fig, clear_figure=True)

    st.caption("Note: This page reports stratified cross-validation results (final comparison).")
    st.stop()


# ------------------------ KNN view ------------------------
if view == "kNN":
    st.header("kNN")

    if knn_rm_df is None and knn_py_df is None:
        st.error("Could not find KNN results (RapidMiner or Python).")
        st.write("Tried RapidMiner:", [os.path.abspath(p) for p in CANDIDATE_KNN_RM])
        st.write("Tried Python:", [os.path.abspath(p) for p in CANDIDATE_KNN_PY])
        st.stop()

    df_any = knn_rm_df if knn_rm_df is not None else knn_py_df
    if "k" not in df_any.columns or "accuracy" not in df_any.columns:
        st.error("KNN file must contain columns: k, accuracy, pred_1_1, pred_1_0, pred_0_1, pred_0_0")
        st.stop()

    k_values = sorted(pd.to_numeric(df_any["k"], errors="coerce").dropna().astype(int).unique().tolist())
    best_k = best_knn_k(df_any)

    if "knn_k" not in st.session_state:
        st.session_state["knn_k"] = best_k

    selected_k = st.sidebar.selectbox(
        "k",
        k_values,
        index=k_values.index(int(st.session_state["knn_k"])) if int(st.session_state["knn_k"]) in k_values else 0,
        key="knn_k",
    )

    rm_row = None
    if knn_rm_df is not None and "k" in knn_rm_df.columns:
        tmp = knn_rm_df[knn_rm_df["k"].astype(int) == int(selected_k)]
        if not tmp.empty:
            rm_row = tmp.iloc[0]

    py_row = None
    if knn_py_df is not None and "k" in knn_py_df.columns:
        tmp = knn_py_df[knn_py_df["k"].astype(int) == int(selected_k)]
        if not tmp.empty:
            py_row = tmp.iloc[0]

    show_dual_results(
        title_top="RapidMiner results",
        row_top=rm_row,
        title_bottom="Python results (same k)",
        row_bottom=py_row,
    )


# ------------------------ RF view ------------------------
elif view == "Random Forest":
    st.header("Random Forest")

    if rf_rm_df is None and rf_py_df is None:
        st.error("Could not find Random Forest results (RapidMiner or Python).")
        st.write("Tried RapidMiner:", [os.path.abspath(p) for p in CANDIDATE_RF_RM])
        st.write("Tried Python:", [os.path.abspath(p) for p in CANDIDATE_RF_PY])
        st.stop()

    df_any = rf_rm_df if rf_rm_df is not None else rf_py_df
    needed = ["number_of_trees", "max_depth", "prepruning", "accuracy"]
    missing = [c for c in needed if c not in df_any.columns]
    if missing:
        st.error(f"RF file is missing columns: {missing}")
        st.stop()

    trees_vals = sorted(pd.to_numeric(df_any["number_of_trees"], errors="coerce").dropna().astype(int).unique().tolist())
    depth_vals = sorted(pd.to_numeric(df_any["max_depth"], errors="coerce").dropna().astype(int).unique().tolist())
    pre_vals = sorted(df_any["prepruning"].astype(str).str.upper().unique().tolist())

    bt, bd, bp = best_rf_config(df_any)

    if "rf_trees" not in st.session_state:
        st.session_state["rf_trees"] = bt
    if "rf_depth" not in st.session_state:
        st.session_state["rf_depth"] = bd
    if "rf_prepruning" not in st.session_state:
        st.session_state["rf_prepruning"] = bp

    selected_trees = st.sidebar.selectbox(
        "number_of_trees",
        trees_vals,
        index=trees_vals.index(int(st.session_state["rf_trees"])) if int(st.session_state["rf_trees"]) in trees_vals else 0,
        key="rf_trees",
    )
    selected_depth = st.sidebar.selectbox(
        "max_depth",
        depth_vals,
        index=depth_vals.index(int(st.session_state["rf_depth"])) if int(st.session_state["rf_depth"]) in depth_vals else 0,
        key="rf_depth",
    )
    selected_pre = st.sidebar.selectbox(
        "prepruning",
        pre_vals,
        index=pre_vals.index(str(st.session_state["rf_prepruning"]).upper()) if str(st.session_state["rf_prepruning"]).upper() in pre_vals else 0,
        key="rf_prepruning",
    )

    rm_row = None
    if rf_rm_df is not None:
        f = rf_rm_df[
            (rf_rm_df["number_of_trees"].astype(int) == int(selected_trees)) &
            (rf_rm_df["max_depth"].astype(int) == int(selected_depth)) &
            (rf_rm_df["prepruning"].astype(str).str.upper() == str(selected_pre).upper())
        ]
        if not f.empty:
            rm_row = f.loc[f["accuracy"].astype(float).idxmax()]

    py_row = None
    if rf_py_df is not None:
        f = rf_py_df[
            (rf_py_df["number_of_trees"].astype(int) == int(selected_trees)) &
            (rf_py_df["max_depth"].astype(int) == int(selected_depth)) &
            (rf_py_df["prepruning"].astype(str).str.upper() == str(selected_pre).upper())
        ]
        if not f.empty:
            py_row = f.loc[f["accuracy"].astype(float).idxmax()]

    show_dual_results(
        title_top="RapidMiner results",
        row_top=rm_row,
        title_bottom="Python results",
        row_bottom=py_row,
    )


# ------------------------ Logistic Regression view ------------------------
elif view == "Logistic Regression":
    st.header("Logistic Regression")

    if lr_rm_df is None and lr_py_df is None:
        st.error("Could not find Logistic Regression results (RapidMiner or Python).")
        st.write("Tried RapidMiner:", [os.path.abspath(p) for p in CANDIDATE_LR_RM])
        st.write("Tried Python:", [os.path.abspath(p) for p in CANDIDATE_LR_PY])
        st.stop()

    rm_row = None
    if lr_rm_df is not None and "accuracy" in lr_rm_df.columns:
        rm_row = lr_rm_df.loc[lr_rm_df["accuracy"].astype(float).idxmax()]

    py_row = None
    if lr_py_df is not None and "accuracy" in lr_py_df.columns:
        py_row = lr_py_df.loc[lr_py_df["accuracy"].astype(float).idxmax()]

    show_dual_results(
        title_top="RapidMiner results",
        row_top=rm_row,
        title_bottom="Python results",
        row_bottom=py_row,
    )

    with st.expander("Show full Logistic Regression tables"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("RapidMiner table")
            if lr_rm_df is not None:
                st.dataframe(lr_rm_df, use_container_width=True)
            else:
                st.info("No RapidMiner LR file found.")
        with c2:
            st.write("Python table")
            if lr_py_df is not None:
                st.dataframe(lr_py_df, use_container_width=True)
            else:
                st.info("No Python LR file found.")


# ------------------------ XGBoost view (CSV results) ------------------------
elif view == "XGBoost":
    st.header("XGBoost")

    if xgb_df is None:
        st.error("Could not find XGBoost results (Python).")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_XGB])
        st.stop()

    # We expect one row (id=1). If multiple rows exist, allow selection.
    if "id" in xgb_df.columns and len(xgb_df) > 1:
        ids = sorted(pd.to_numeric(xgb_df["id"], errors="coerce").dropna().astype(int).unique().tolist())
        selected_id = st.sidebar.selectbox("id", ids, index=0, key="xgb_id")
        tmp = xgb_df[xgb_df["id"].astype(int) == int(selected_id)]
        row = tmp.iloc[0] if not tmp.empty else xgb_df.iloc[0]
    else:
        row = xgb_df.iloc[0]

    show_dual_results(
        title_top="Python results",
        row_top=row,
        title_bottom="",
        row_bottom=None,
    )

    with st.expander("Model hyperparameters (if available)"):
        params = {}
        for k in ["n_estimators", "max_depth", "learning_rate"]:
            if k in xgb_df.columns:
                params[k] = row[k]
        if params:
            st.write(params)
        else:
            st.info("No hyperparameter columns found in xgb_results.csv.")


# ------------------------ Sample data view ------------------------
else:
    st.header("Sample data")

    if sample_df is None:
        st.error("Could not find investments_VC file in Data/")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_SAMPLE])
        st.stop()

    df = sample_df.copy()

    # --- STRICT FILTER: only acquired / closed ---
    if "status" not in df.columns:
        st.error("Dataset must contain column: status")
        st.stop()

    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df = df[df["status"].isin(["acquired", "closed"])]

    # ---------------- Sidebar filters ----------------
    st.sidebar.subheader("Sample data filters")

    n_rows = st.sidebar.slider(
        "Rows to display",
        min_value=5,
        max_value=200,
        value=20,
        step=5
    )

    if "market" in df.columns:
        markets = sorted(df["market"].dropna().astype(str).unique().tolist())
        market_choice = st.sidebar.selectbox("market", ["All"] + markets)
        if market_choice != "All":
            df = df[df["market"].astype(str) == market_choice]

    status_choice = st.sidebar.selectbox(
        "status",
        ["All", "acquired", "closed"]
    )
    if status_choice != "All":
        df = df[df["status"] == status_choice]

    # ---------------- Display ----------------
    display_df = df.head(n_rows).copy()

    if "funding_total_usd" in display_df.columns:
        display_df["funding_total_usd"] = display_df["funding_total_usd"].apply(format_money)

    preferred_order = ["funding_total_usd", "market", "funding_rounds", "status", "firm_age"]
    cols = display_df.columns.tolist()
    ordered = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    display_df = display_df[ordered]

    st.dataframe(display_df, use_container_width=True)

    # ---------------- Summary ----------------
    with st.expander("Quick summary"):
        st.write(f"Rows after filters: {len(df)}")
        st.write("Status distribution:")
        st.dataframe(df["status"].value_counts().to_frame("count"))
        if market_choice == "All" and "market" in df.columns:
            st.write("Market distribution:")
            st.dataframe(
                df["market"]
                .value_counts()
                .to_frame("count")
            )
        country_col = None
        for c in ["country", "country_code", "country_name"]:
            if c in df.columns:
                country_col = c
                break

        if country_col is not None:
            st.write("Country distribution:")
            st.dataframe(
                df[country_col]
                .value_counts()
                .head(15)
                .to_frame("count"))

