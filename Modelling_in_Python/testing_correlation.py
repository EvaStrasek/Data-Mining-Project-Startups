import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# --------------------------------------------------
# Load data
# --------------------------------------------------
FILE = "../Data/startups_new.csv"   # <- zamenjaj z dejanskim imenom
df = pd.read_csv(FILE, sep=";", engine="python", encoding="latin1")
df.columns = df.columns.str.strip()

print("Rows before cleaning:", len(df))

# --------------------------------------------------
# Target: acquired vs closed
# --------------------------------------------------
df["target"] = df["status"].map({
    "acquired": 1,
    "closed": 0
})

df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

print("Rows after filtering:", len(df))
print("Target distribution:\n", df["target"].value_counts())

# --------------------------------------------------
# Numeric features → Spearman correlation
# --------------------------------------------------
num_features = [
    "funding_total_usd",
    "funding_rounds",
    "seed", "venture", "equity_crowdfunding", "undisclosed",
    "convertible_note", "debt_financing", "angel", "grant",
    "private_equity", "post_ipo_equity", "post_ipo_debt",
    "secondary_market", "product_crowdfunding",
    "round_A", "round_B", "round_C", "round_D",
    "round_E", "round_F", "round_G", "round_H",
    "founded_year"
]

for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

num_df = df[num_features + ["target"]].dropna()

print("\n=== SPEARMAN CORRELATION (numeric vs acquired=1) ===\n")
spearman = num_df.corr(method="spearman")["target"].sort_values()
print(spearman)

# --------------------------------------------------
# Binary category features → Chi-square
# --------------------------------------------------
bin_features = [
    "country_code", "state_code", "region",
    "market", "category_list"
]

print("\n=== CHI-SQUARE TESTS (categorical vs acquired) ===\n")

for col in bin_features:
    if col not in df.columns:
        continue

    # reduce high-cardinality columns
    top = df[col].value_counts().nlargest(10).index
    df_sub = df[df[col].isin(top)]

    table = pd.crosstab(df_sub[col], df_sub["target"])

    if table.shape[0] < 2:
        continue

    chi2, p, dof, exp = chi2_contingency(table)

    # Cramer's V
    n = table.sum().sum()
    r, k = table.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

    print(f"{col:12s}  p-value = {p:.4g}   Cramer's V = {v:.3f}")

print("\nDONE")
