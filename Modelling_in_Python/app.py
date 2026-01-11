import os
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------ Robust base dir ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ------------------------ File paths ------------------------
CANDIDATE_KNN = [
    os.path.join(BASE_DIR, "Data", "results_knn.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_knn.csv"),
]
CANDIDATE_RF = [
    os.path.join(BASE_DIR, "Data", "results_random_forest.csv"),
    os.path.join(BASE_DIR, "..", "Data", "results_random_forest.csv"),
]
CANDIDATE_SAMPLE = [
    os.path.join(BASE_DIR, "Data", "startups_python.csv"),
    os.path.join(BASE_DIR, "..", "Data", "startups_python.csv"),
]

KNN_PATH = first_existing(CANDIDATE_KNN)
RF_PATH = first_existing(CANDIDATE_RF)
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

    for col in [
        "accuracy",
        "class_recall_1_1",
        "class_recall_0_0",
        "class_precision_1_1",
        "class_precision_0_0",
    ]:
        if col in df.columns:
            df[col] = pct_to_float(df[col])

    for col in ["pred_1_1", "pred_1_0", "pred_0_1", "pred_0_0"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def rapidminer_table(row):
    # TP/FN/FP/TN mapping:
    TP = int(row["pred_1_1"])
    FN = int(row["pred_1_0"])
    FP = int(row["pred_0_1"])
    TN = int(row["pred_0_0"])

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0

    prec_pred_false = TN / (TN + FN) if (TN + FN) else 0.0
    prec_pred_true = TP / (TP + FP) if (TP + FP) else 0.0
    rec_false = TN / (TN + FP) if (TN + FP) else 0.0
    rec_true = TP / (TP + FN) if (TP + FN) else 0.0

    table = pd.DataFrame(
        {
            "true false": [TN, FP, f"{rec_false*100:.2f}%"],
            "true true": [FN, TP, f"{rec_true*100:.2f}%"],
            "class precision": [f"{prec_pred_false*100:.2f}%", f"{prec_pred_true*100:.2f}%", ""],
        },
        index=["pred. false", "pred. true", "class recall"],
    )
    return acc, table


def best_knn_k(knn_df: pd.DataFrame) -> int:
    best_row = knn_df.loc[knn_df["accuracy"].astype(float).idxmax()]
    return int(best_row["k"])


def best_rf_config(rf_df: pd.DataFrame) -> tuple[int, int, str]:
    best_row = rf_df.loc[rf_df["accuracy"].astype(float).idxmax()]
    trees = int(best_row["number_of_trees"])
    depth = int(best_row["max_depth"])
    pruning = str(best_row["pruning"]).strip().upper()
    return trees, depth, pruning


def load_sample_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    # Strip whitespace in market if present
    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.strip()

    # Convert obvious numeric columns to numeric (handles 3.50E+07 etc.)
    numeric_cols = [c for c in df.columns if c != "market"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Force numeric for funding_total_usd if exists
    if "funding_total_usd" in df.columns:
        df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

    # Status + firm_age numeric if exist
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


# ------------------------ Streamlit UI ------------------------
st.set_page_config(layout="wide")
st.title("Startups prediction and modelling")

# Load KNN/RF if available (we’ll only error when user selects them)
knn_df = load_results(KNN_PATH) if KNN_PATH else None
rf_df = load_results(RF_PATH) if RF_PATH else None
if rf_df is not None and "pruning" in rf_df.columns:
    rf_df["pruning"] = rf_df["pruning"].astype(str).str.strip().str.upper()

sample_df = load_sample_dataset(SAMPLE_PATH) if SAMPLE_PATH else None


# Sidebar
st.sidebar.header("Navigation")

view = st.sidebar.radio(
    "Choose view",
    ["Sample data","kNN", "Random Forest"],
    key="view_choice",
)

st.sidebar.divider()

# Reset button (for KNN/RF selections)
if st.sidebar.button("Reset to best accuracy"):
    if knn_df is not None and "k" in knn_df.columns and "accuracy" in knn_df.columns:
        st.session_state["knn_k"] = best_knn_k(knn_df)

    if rf_df is not None and all(c in rf_df.columns for c in ["number_of_trees", "max_depth", "pruning", "accuracy"]):
        bt, bd, bp = best_rf_config(rf_df)
        st.session_state["rf_trees"] = bt
        st.session_state["rf_depth"] = bd
        st.session_state["rf_pruning"] = bp

    # optional: keep current view, just reset params
    st.rerun()


# ------------------------ KNN view ------------------------
if view == "KNN":
    st.header("kNN")

    if knn_df is None:
        st.error("Could not find results_knn.csv")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_KNN])
        st.stop()

    if "k" not in knn_df.columns or "accuracy" not in knn_df.columns:
        st.error("KNN file must contain columns: k, accuracy, pred_1_1, pred_1_0, pred_0_1, pred_0_0")
        st.stop()

    k_values = sorted(pd.to_numeric(knn_df["k"], errors="coerce").dropna().astype(int).unique().tolist())
    best_k = best_knn_k(knn_df)

    if "knn_k" not in st.session_state:
        st.session_state["knn_k"] = best_k

    selected_k = st.sidebar.selectbox(
        "k",
        k_values,
        index=k_values.index(int(st.session_state["knn_k"])) if int(st.session_state["knn_k"]) in k_values else 0,
        key="knn_k",
    )

    row = knn_df[knn_df["k"].astype(int) == int(selected_k)].iloc[0]
    acc, table = rapidminer_table(row)

    st.subheader(f"accuracy: {acc*100:.2f}%")
    st.dataframe(table, use_container_width=True)


# ------------------------ RF view ------------------------
elif view == "Random Forest":
    st.header("Random Forest")

    if rf_df is None:
        st.error("Could not find results_random_forest.csv")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_RF])
        st.stop()

    needed = ["number_of_trees", "max_depth", "pruning", "accuracy"]
    missing = [c for c in needed if c not in rf_df.columns]
    if missing:
        st.error(f"RF file is missing columns: {missing}")
        st.stop()

    trees_vals = sorted(pd.to_numeric(rf_df["number_of_trees"], errors="coerce").dropna().astype(int).unique().tolist())
    depth_vals = sorted(pd.to_numeric(rf_df["max_depth"], errors="coerce").dropna().astype(int).unique().tolist())
    prune_vals = sorted(rf_df["pruning"].astype(str).str.upper().unique().tolist())

    bt, bd, bp = best_rf_config(rf_df)

    if "rf_trees" not in st.session_state:
        st.session_state["rf_trees"] = bt
    if "rf_depth" not in st.session_state:
        st.session_state["rf_depth"] = bd
    if "rf_pruning" not in st.session_state:
        st.session_state["rf_pruning"] = bp

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
    selected_pruning = st.sidebar.selectbox(
        "prepruning",
        prune_vals,
        index=prune_vals.index(str(st.session_state["rf_pruning"]).upper()) if str(st.session_state["rf_pruning"]).upper() in prune_vals else 0,
        key="rf_pruning",
    )

    filtered = rf_df[
        (rf_df["number_of_trees"].astype(int) == int(selected_trees))
        & (rf_df["max_depth"].astype(int) == int(selected_depth))
        & (rf_df["pruning"].astype(str).str.upper() == str(selected_pruning).upper())
    ]

    if filtered.empty:
        st.warning("No run matches this combination.")
        st.stop()

    row = filtered.loc[filtered["accuracy"].astype(float).idxmax()]
    acc, table = rapidminer_table(row)

    st.subheader(f"accuracy: {acc*100:.2f}%")
    st.dataframe(table, use_container_width=True)


# ------------------------ Sample data view ------------------------
else:
    st.header("Sample data")

    if sample_df is None:
        st.error("Could not find startups_python file in Data/")
        st.write("Tried:", [os.path.abspath(p) for p in CANDIDATE_SAMPLE])
        st.stop()

    # Sidebar filters
    st.sidebar.subheader("Sample data filters")

    n_rows = st.sidebar.slider("Rows to display", min_value=5, max_value=200, value=20, step=5)

    df = sample_df.copy()

    # Market filter
    if "market" in df.columns:
        markets = sorted(df["market"].dropna().astype(str).unique().tolist())
        market_choice = st.sidebar.selectbox("market", ["All"] + markets)
        if market_choice != "All":
            df = df[df["market"].astype(str) == market_choice]

    # Status filter
    if "Status" in df.columns:
        statuses = sorted(df["Status"].dropna().astype(int).unique().tolist())
        status_choice = st.sidebar.selectbox("Status", ["All"] + statuses)
        if status_choice != "All":
            df = df[df["Status"].astype(int) == int(status_choice)]

    # Build a prettier display DF
    display_df = df.head(n_rows).copy()

    # Pretty money formatting for funding_total_usd
    if "funding_total_usd" in display_df.columns:
        display_df["funding_total_usd"] = display_df["funding_total_usd"].apply(format_money)

    # Put key columns first if they exist
    preferred_order = ["funding_total_usd", "market", "funding_rounds", "Status", "firm_age"]
    cols = display_df.columns.tolist()
    ordered = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    display_df = display_df[ordered]

    st.caption(f"Loaded from: {SAMPLE_PATH}")
    st.dataframe(display_df, use_container_width=True)

    # Optional small summary (clean, not “all results”)
    with st.expander("Quick summary"):
        st.write(f"Rows after filters: {len(df)}")
        if "Status" in df.columns:
            st.write("Status distribution:")
            st.dataframe(df["Status"].value_counts(dropna=False).to_frame("count"))
        if "market" in df.columns:
            st.write("Top markets:")
            st.dataframe(df["market"].value_counts(dropna=False).head(10).to_frame("count"))
