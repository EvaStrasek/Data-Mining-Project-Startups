import os
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------ Robust base dir ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


# ------------------------ File paths ------------------------
CANDIDATE_KNN_RM = [
    os.path.join(BASE_DIR, "Data", "results_knn.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_knn.csv"),
]
CANDIDATE_KNN_PY = [
    os.path.join(BASE_DIR, "Data", "knn_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "knn_results.csv"),
    os.path.join(BASE_DIR, "Data", "results_knn_python.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_knn_python.csv"),
]

CANDIDATE_RF_RM = [
    os.path.join(BASE_DIR, "Data", "results_random_forest.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_random_forest.csv"),
]
CANDIDATE_RF_PY = [
    os.path.join(BASE_DIR, "Data", "rf_results.csv"),
    os.path.join(BASE_DIR, "..", "Data", "rf_results.csv"),
    os.path.join(BASE_DIR, "Data", "results_random_forest_python.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_random_forest_python.csv"),
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

CANDIDATE_SAMPLE = [
    os.path.join(BASE_DIR, "Data", "startups_python.csv"),
    os.path.join(BASE_DIR, "..", "Data", "startups_python.csv"),
]

KNN_RM_PATH = first_existing(CANDIDATE_KNN_RM)
KNN_PY_PATH = first_existing(CANDIDATE_KNN_PY)

RF_RM_PATH = first_existing(CANDIDATE_RF_RM)
RF_PY_PATH = first_existing(CANDIDATE_RF_PY)

LR_RM_PATH = first_existing(CANDIDATE_LR_RM)
LR_PY_PATH = first_existing(CANDIDATE_LR_PY)

SAMPLE_PATH = first_existing(CANDIDATE_SAMPLE)


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
    df = pd.read_csv(path, sep=";")
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


# ------------------------ Streamlit UI ------------------------
st.set_page_config(layout="wide")
st.title("Startups prediction and modelling")

# Load datasets (if they exist)
knn_rm_df = load_results(KNN_RM_PATH) if KNN_RM_PATH else None
knn_py_df = load_results(KNN_PY_PATH) if KNN_PY_PATH else None

rf_rm_df = load_results(RF_RM_PATH) if RF_RM_PATH else None
rf_py_df = load_results(RF_PY_PATH) if RF_PY_PATH else None

lr_rm_df = load_results(LR_RM_PATH) if LR_RM_PATH else None
lr_py_df = load_results(LR_PY_PATH) if LR_PY_PATH else None

sample_df = load_sample_dataset(SAMPLE_PATH) if SAMPLE_PATH else None


# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Choose view",
    ["Sample data", "kNN", "Random Forest", "Logistic Regression"],
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


# ------------------------ Sample data view ------------------------
else:
    st.header("Sample data")

    if sample_df is None:
        st.error("Could not find startups_python file in Data/")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_SAMPLE])
        st.stop()

    st.sidebar.subheader("Sample data filters")
    n_rows = st.sidebar.slider("Rows to display", min_value=5, max_value=200, value=20, step=5)

    df = sample_df.copy()

    if "market" in df.columns:
        markets = sorted(df["market"].dropna().astype(str).unique().tolist())
        market_choice = st.sidebar.selectbox("market", ["All"] + markets)
        if market_choice != "All":
            df = df[df["market"].astype(str) == market_choice]

    if "Status" in df.columns:
        statuses = sorted(df["Status"].dropna().astype(int).unique().tolist())
        status_choice = st.sidebar.selectbox("Status", ["All"] + statuses)
        if status_choice != "All":
            df = df[df["Status"].astype(int) == int(status_choice)]

    display_df = df.head(n_rows).copy()

    if "funding_total_usd" in display_df.columns:
        display_df["funding_total_usd"] = display_df["funding_total_usd"].apply(format_money)

    preferred_order = ["funding_total_usd", "market", "funding_rounds", "Status", "firm_age"]
    cols = display_df.columns.tolist()
    ordered = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    display_df = display_df[ordered]

    st.caption(f"Loaded from: {SAMPLE_PATH}")
    st.dataframe(display_df, use_container_width=True)

    with st.expander("Quick summary"):
        st.write(f"Rows after filters: {len(df)}")
        if "Status" in df.columns:
            st.write("Status distribution:")
            st.dataframe(df["Status"].value_counts(dropna=False).to_frame("count"))
        if "market" in df.columns:
            st.write("Top markets:")
            st.dataframe(df["market"].value_counts(dropna=False).head(10).to_frame("count"))
