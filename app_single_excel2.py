import io
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# EXCLUDE_COLS is computed after loading the dataset.
EXCLUDE_COLS = set()


# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Survey Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Survey Dashboard (Single Excel)")

st.markdown(
    """
Ανεβάζεις **1 αρχείο Excel** που περιέχει:
- **Data sheet** με τα δεδομένα (στήλες = μεταβλητές)
- (προαιρετικά) **Meta sheet** *μέσα στο ίδιο Excel* με:
  - είτε **2 στήλες**: **Label**, **Category** (το Label πρέπει να είναι ίδιο με το όνομα της στήλης στα δεδομένα)
  - είτε **3 στήλες**: **Code**, **Label**, **Category**
  - π.χ. `A1`, `Dem03`, ...

Το app ανιχνεύει αυτόματα πιθανό meta sheet, αλλά μπορείς να το επιλέξεις χειροκίνητα.

Μετά μπορείς να κάνεις:
- **Απλές κατανομές**
- **Διασταυρώσεις**
- **Κλίμακες στάσεων (PCA + varimax)**
"""
)


# =========================
# Helpers: Meta
# =========================
@dataclass
class MetaSpec:
    # Supports either 3-column meta (code/text/category) or 2-column meta (label/category)
    mode: str  # "3col" or "2col"
    col1: str
    col2: str
    col3: Optional[str] = None


def _guess_meta_spec(meta_df: pd.DataFrame) -> MetaSpec:
    """Guess meta columns.
    Supported formats inside the SAME Excel:
      - 3 columns: code + label + category
      - 2 columns: label + category (label must match the data column names)
    """
    cols = list(meta_df.columns)
    norm = {c: re.sub(r"\s+", "", str(c)).lower() for c in cols}

    def pick(candidates):
        for cand in candidates:
            for c in cols:
                if norm.get(c) == cand:
                    return c
        return None

    # Common headers (Greek/English)
    code = pick(["κωδικός", "κωδικος", "code", "variable", "var", "column"])
    text = pick(["εξήγηση", "εξηγηση", "label", "question", "text", "description"])
    cat  = pick(["κατηγορία", "κατηγορια", "category", "group", "section"])

    # Prefer explicit 3-col format
    if code and text and cat:
        return MetaSpec("3col", code, text, cat)

    # If 2 columns, try label + category
    if text and cat and code is None:
        return MetaSpec("2col", text, cat)

    # Fallbacks by column count
    if len(cols) >= 3:
        return MetaSpec("3col", cols[0], cols[1], cols[2])
    if len(cols) == 2:
        return MetaSpec("2col", cols[0], cols[1])

    raise ValueError("Το meta sheet πρέπει να έχει 2 (Label, Category) ή 3 (Code, Label, Category) στήλες.")


def _looks_like_meta_sheet(df_head: pd.DataFrame) -> bool:
    """Heuristic: meta sheet usually has label/category-like column names.
    Supports 2-col or 3-col dictionaries.
    """
    cols = [str(c).strip().lower() for c in df_head.columns]
    joined = " ".join(cols)
    keywords = [
        "κωδ", "code", "var", "variable",
        "εξηγ", "label", "question", "descr",
        "κατηγ", "category", "section",
    ]
    hits = sum(1 for k in keywords if k in joined)
    return (df_head.shape[1] >= 2) and (hits >= 2)


def _default_sheet_choices(xls: pd.ExcelFile) -> Tuple[str, Optional[str]]:
    """
    Guess:
      - data_sheet: prefer common names; else first non-meta; else first sheet
      - meta_sheet: prefer common names or meta-like heuristic; else None
    """
    sheet_names = xls.sheet_names
    meta_candidates = []
    for sh in sheet_names:
        try:
            head = pd.read_excel(xls, sheet_name=sh, nrows=5)
            if _looks_like_meta_sheet(head):
                meta_candidates.append(sh)
        except Exception:
            continue

    preferred_meta = None
    for nm in sheet_names:
        if str(nm).strip().lower() in {"meta", "metadata", "questions", "questionnaire", "dictionary", "labels"}:
            preferred_meta = nm
            break
    if preferred_meta is None and meta_candidates:
        preferred_meta = meta_candidates[0]

    preferred_data = None
    for nm in sheet_names:
        if str(nm).strip().lower() in {"data", "dataset", "responses", "sheet1"}:
            preferred_data = nm
            break
    if preferred_data is None:
        non_meta = [sh for sh in sheet_names if sh != preferred_meta]
        preferred_data = non_meta[0] if non_meta else sheet_names[0]

    return preferred_data, preferred_meta


@st.cache_data(show_spinner=False)
def load_excel(uploaded_file, data_sheet: str, meta_sheet: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[MetaSpec]]:
    """Load dataset and optional meta from the same Excel file.

    Meta formats supported:
      1) 3 columns: Code, Label, Category
      2) 2 columns: Label, Category (Label must match the DATA column names)
    """
    df = pd.read_excel(uploaded_file, sheet_name=data_sheet)

    meta = None
    spec = None
    if meta_sheet and meta_sheet != "(None)":
        meta_raw = pd.read_excel(uploaded_file, sheet_name=meta_sheet)
        spec = _guess_meta_spec(meta_raw)

        if spec.mode == "3col":
            meta = meta_raw[[spec.col1, spec.col2, spec.col3]].copy()
            meta.columns = ["code", "text", "category"]
            meta["code"] = meta["code"].astype(str)
            meta["text"] = meta["text"].astype(str)
            meta["category"] = meta["category"].astype(str)

        elif spec.mode == "2col":
            meta = meta_raw[[spec.col1, spec.col2]].copy()
            meta.columns = ["text", "category"]
            meta["text"] = meta["text"].astype(str)
            meta["category"] = meta["category"].astype(str)
            # In 2-col mode, the label IS the data column name
            meta["code"] = meta["text"].astype(str)
            meta = meta[["code", "text", "category"]]

        else:
            raise ValueError("Άγνωστο meta format.")

    return df, meta, spec


def build_variable_catalog(df: pd.DataFrame, meta: Optional[pd.DataFrame], exclude_cols: Optional[list] = None) -> pd.DataFrame:
    """Return catalog with columns: code, text, category.

    If meta is None: text=code and category='Όλα'.
    If meta is provided:
      - 3-col: code/text/category as given
      - 2-col: meta.text matches the DATA column names; we use that as code and label.
    """
    cols = list(df.columns)
    exclude_cols = set(exclude_cols or [])

    if meta is None:
        cat = pd.DataFrame({"code": cols, "text": cols, "category": ["Όλα"] * len(cols)})
    else:
        meta2 = meta.copy()

        for required in ["code", "text", "category"]:
            if required not in meta2.columns:
                raise ValueError(f"Meta sheet missing required column: {required}")

        missing = sorted(set(cols) - set(meta2["code"]))
        if missing:
            add = pd.DataFrame({"code": missing, "text": missing, "category": ["(Χωρίς κατηγορία)"] * len(missing)})
            meta2 = pd.concat([meta2, add], ignore_index=True)

        cat = meta2[meta2["code"].isin(cols)].drop_duplicates(subset=["code"], keep="first").copy()
        cat["text"] = cat["text"].fillna(cat["code"]).astype(str)

    return cat[~cat["code"].isin(exclude_cols)].copy()


def select_variable_ui(catalog: pd.DataFrame, key: str, title: str):
    """UI helper to select a variable safely (no string parsing)."""
    st.subheader(title)

    cats = ["Όλα"] + sorted([c for c in catalog["category"].dropna().unique() if c != "Όλα"])
    cat = st.selectbox("Κατηγορία", cats, key=f"{key}_cat")

    subset = catalog if cat == "Όλα" else catalog[catalog["category"] == cat]
    subset = subset.dropna(subset=["code"]).copy()

    if subset.empty:
        st.warning("Δεν υπάρχουν μεταβλητές για αυτήν την κατηγορία. Επιστρέφω σε 'Όλα'.")
        subset = catalog.dropna(subset=["code"]).copy()

    subset = subset.sort_values(["text", "code"]).reset_index(drop=True)

    # Use codes as the real selectbox values; display nice labels via format_func.
    code_to_label = {r["code"]: f"{r['text']}  [{r['code']}]" for _, r in subset.iterrows()}
    codes = subset["code"].tolist()

    chosen_code = st.selectbox(
        "Μεταβλητή",
        options=codes,
        key=f"{key}_var",
        format_func=lambda c: code_to_label.get(c, str(c))
    )

    row = subset[subset["code"] == chosen_code]
    if row.empty:
        row = catalog[catalog["code"] == chosen_code]
    if row.empty:
        return chosen_code, str(chosen_code), cat

    row = row.iloc[0]
    return row["code"], row["text"], row["category"]


# =========================
# Helpers: Filtering / Weighting
# =========================
DEFAULT_NA_STRINGS = {
    "Δε γνωρίζω / Δεν απαντώ",
    "Δε γνωρίζω/Δεν απαντώ",
    "Δεν απαντώ",
    "No answer",
    "Refuse",
    "99",
    "NA",
}
DEFAULT_DK_STRINGS = {
    "Δεν ξέρω",
    "Δε γνωρίζω",
    "Don't know",
    "DK",
    "88",
    "Μη εφαρμόσιμο",
    "Μη εφαρμοσιμο",
}

def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    sample = pd.to_numeric(s.dropna().head(50), errors="coerce")
    return sample.notna().all() and len(sample) > 0


def apply_missing_filters(s: pd.Series, drop_na99: bool, drop_dk88: bool, na_strings: set, dk_strings: set) -> pd.Series:
    s2 = s.copy()

    if not _is_numeric_series(s2):
        s2 = s2.astype("string").str.strip()

    mask = pd.Series(True, index=s2.index)

    if drop_na99:
        mask &= s2.notna()
        if _is_numeric_series(s2):
            s_num = pd.to_numeric(s2, errors="coerce")
            mask &= (s_num != 99)
        else:
            mask &= ~s2.isin(list(na_strings))

    if drop_dk88:
        if _is_numeric_series(s2):
            s_num = pd.to_numeric(s2, errors="coerce")
            mask &= (s_num != 88)
        else:
            mask &= ~s2.isin(list(dk_strings))

    return s2[mask]


def weighted_counts(s: pd.Series, w: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"answer": s, "w": w}).dropna()
    out = tmp.groupby("answer")["w"].sum().sort_values(ascending=False).reset_index()
    out.columns = ["answer", "count"]
    out["percent"] = out["count"] / out["count"].sum() * 100
    return out


def unweighted_counts(s: pd.Series) -> pd.DataFrame:
    out = s.value_counts(dropna=False).reset_index()
    out.columns = ["answer", "count"]
    out["percent"] = out["count"] / out["count"].sum() * 100
    return out


# =========================
# Charts
# =========================
SUPPORTED_CHARTS_UNIVAR = [
    "Bar (vertical)",
    "Bar (horizontal)",
    "Pie",
    "Donut",
    "Dot plot",
    "Connected dot plot (lollipop)",
    "Treemap",
    "Sunburst",
    "Waffle (approx)",
    "Polar bar",
]

SUPPORTED_CHARTS_BIVAR = [
    "Stacked bar (row %)",
    "Stacked bar (counts)",
    "Heatmap (row %)",
    "Heatmap (counts)",
]

def plot_univariate(freq: pd.DataFrame, chart_type: str, title: str):
    d = freq.copy()
    d["answer"] = d["answer"].astype(str)

    if chart_type == "Bar (vertical)":
        fig = px.bar(d, x="answer", y="percent", title=title)
        fig.update_layout(xaxis_title="", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar (horizontal)":
        fig = px.bar(d, x="percent", y="answer", orientation="h", title=title)
        fig.update_layout(xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie":
        fig = px.pie(d, names="answer", values="percent", title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Donut":
        fig = px.pie(d, names="answer", values="percent", title=title, hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Dot plot":
        fig = px.scatter(d, x="percent", y="answer", title=title)
        fig.update_layout(xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Connected dot plot (lollipop)":
        fig = go.Figure()
        for _, r in d.iterrows():
            fig.add_trace(go.Scatter(x=[0, r["percent"]], y=[str(r["answer"]), str(r["answer"])], mode="lines", showlegend=False))
            fig.add_trace(go.Scatter(x=[r["percent"]], y=[str(r["answer"])], mode="markers", showlegend=False))
        fig.update_layout(title=title, xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Treemap":
        fig = px.treemap(d, path=["answer"], values="percent", title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Sunburst":
        fig = px.sunburst(d, path=["answer"], values="percent", title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Waffle (approx)":
        top = d.sort_values("percent", ascending=False).head(10)
        tiles = []
        for _, r in top.iterrows():
            tiles += [r["answer"]] * int(round(r["percent"]))
        tiles = tiles[:100] + ["(λοιπά)"] * max(0, 100 - len(tiles))
        grid = np.array(tiles).reshape(10, 10)
        fig = px.imshow(grid, aspect="equal", title=title)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Polar bar":
        fig = px.bar_polar(d, r="percent", theta="answer", title=title)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Αυτός ο τύπος διαγράμματος δεν έχει υλοποιηθεί ακόμη.")


def plot_bivariate(ct: pd.DataFrame, chart_type: str, title: str):
    if chart_type == "Stacked bar (counts)":
        long = ct.reset_index().melt(id_vars=ct.index.name or "row", var_name="col", value_name="value")
        row_name = ct.index.name or "row"
        long.columns = [row_name, "col", "value"]
        fig = px.bar(long, x=row_name, y="value", color="col", title=title, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Stacked bar (row %)":
        rowp = ct.div(ct.sum(axis=1), axis=0) * 100
        long = rowp.reset_index().melt(id_vars=rowp.index.name or "row", var_name="col", value_name="value")
        row_name = rowp.index.name or "row"
        long.columns = [row_name, "col", "value"]
        fig = px.bar(long, x=row_name, y="value", color="col", title=title, barmode="stack")
        fig.update_layout(yaxis_title="% (row)")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap (counts)":
        fig = px.imshow(ct, aspect="auto", title=title)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap (row %)":
        rowp = ct.div(ct.sum(axis=1), axis=0) * 100
        fig = px.imshow(rowp.round(2), aspect="auto", title=title)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Αυτός ο τύπος διαγράμματος δεν έχει υλοποιηθεί ακόμη.")


# =========================
# PCA / Varimax / Reliability
# =========================
def varimax(Phi, gamma=1.0, q=30, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


def cronbach_alpha(X: np.ndarray) -> float:
    if X.shape[1] < 2:
        return np.nan
    X = X.astype(float)
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    p = X.shape[1]
    return (p / (p - 1)) * (1 - item_vars.sum() / total_var)


def kmo_bartlett(X: np.ndarray):
    X = X.astype(float)
    R = np.corrcoef(X, rowvar=False)
    invR = np.linalg.pinv(R)

    p = R.shape[0]
    A = np.zeros_like(R)
    for i in range(p):
        for j in range(p):
            A[i, j] = -invR[i, j] / math.sqrt(invR[i, i] * invR[j, j])
    np.fill_diagonal(A, 0)

    r2 = R**2
    a2 = A**2
    np.fill_diagonal(r2, 0)

    kmo_num = r2.sum()
    kmo_den = kmo_num + a2.sum()
    kmo = kmo_num / kmo_den if kmo_den != 0 else np.nan

    n = X.shape[0]
    detR = max(np.linalg.det(R), 1e-12)
    chi2 = -(n - 1 - (2*p + 5)/6) * math.log(detR)
    dof = p*(p-1)/2
    pval = 1 - stats.chi2.cdf(chi2, dof)
    return float(kmo), float(chi2), float(pval)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("1) Upload")
    xlsx_file = st.file_uploader("Excel (δεδομένα + προαιρετικά meta sheet)", type=["xlsx", "xls"])

    st.divider()
    st.header("2) Options")
    use_weights = st.checkbox("Χρήση στάθμισης", value=True)
    drop_na99 = st.checkbox("Αφαίρεση κενών + '99' (Δεν απαντώ)", value=True)
    drop_dk88 = st.checkbox("Αφαίρεση '88' (Δεν ξέρω / Μη εφαρμόσιμο)", value=False)
    st.caption("Σε λεκτικά δεδομένα, το app αντιστοιχίζει 99/88 σε κοινές φράσεις.")
    na_extra = st.text_input("Extra NA strings (comma-separated)", value="")
    dk_extra = st.text_input("Extra DK strings (comma-separated)", value="")

    st.divider()
    st.header("3) Weight column")
    weight_col_name = st.text_input("Όνομα στήλης στάθμισης", value="weight")


if xlsx_file is None:
    st.info("⬅️ Ανέβασε πρώτα ένα Excel για να ξεκινήσεις.")
    st.stop()

xls = pd.ExcelFile(xlsx_file)
data_default, meta_default = _default_sheet_choices(xls)

with st.sidebar:
    st.divider()
    st.header("4) Sheets")
    data_sheet = st.selectbox("Data sheet", options=xls.sheet_names, index=xls.sheet_names.index(data_default))
    meta_options = ["(None)"] + xls.sheet_names
    meta_idx = meta_options.index(meta_default) if meta_default in meta_options else 0
    meta_sheet = st.selectbox("Meta sheet (optional)", options=meta_options, index=meta_idx)
    st.caption("Αν δεν υπάρχει meta sheet, οι περιγραφές = ονόματα στηλών.")

df, meta, _spec = load_excel(xlsx_file, data_sheet=data_sheet, meta_sheet=meta_sheet)

# === Auto-exclude very high-cardinality columns (e.g. ID) from dropdowns to keep the UI fast ===
EXCLUDE_COLS = set()

# 1) Common ID names
for c in df.columns:
    if str(c).strip().lower() in {"id", "respondent_id", "response_id"}:
        EXCLUDE_COLS.add(c)

# 2) High-cardinality heuristic: almost unique per row
n_rows = max(1, len(df))
for c in df.columns:
    cname = str(c).strip().lower()
    if cname in {"weight", "w", "weights"}:
        continue
    try:
        nunq = df[c].nunique(dropna=True)
        if nunq / n_rows >= 0.90:
            EXCLUDE_COLS.add(c)
    except Exception:
        pass

with st.sidebar:
    st.caption(f"Αυτόματη εξαίρεση από dropdowns: {len(EXCLUDE_COLS)} στήλες (π.χ. ID/σχεδόν μοναδικές).")

catalog = build_variable_catalog(df, meta, exclude_cols=sorted(EXCLUDE_COLS))

with st.expander("Preview δεδομένων (πρώτες 20 γραμμές)", expanded=False):
    preview_df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors="ignore")
    st.dataframe(preview_df.head(20), use_container_width=True)

# Weights
if use_weights and weight_col_name in df.columns:
    w = pd.to_numeric(df[weight_col_name], errors="coerce").fillna(0.0)
else:
    w = pd.Series(1.0, index=df.index)

na_strings = set(DEFAULT_NA_STRINGS)
dk_strings = set(DEFAULT_DK_STRINGS)
if na_extra.strip():
    na_strings |= {x.strip() for x in na_extra.split(",") if x.strip()}
if dk_extra.strip():
    dk_strings |= {x.strip() for x in dk_extra.split(",") if x.strip()}

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Απλές κατανομές", "Διασταυρώσεις", "Κλίμακες στάσεων (PCA)"])


# -------- Tab 1 --------
with tab1:
    colA, colB = st.columns([1.1, 1.0], gap="large")
    with colA:
        code, text, cat = select_variable_ui(catalog, key="uni", title="Απλή κατανομή")
        st.caption(f"Επιλεγμένη μεταβλητή: **{code}** — {text} ({cat})")

        s = apply_missing_filters(df[code], drop_na99, drop_dk88, na_strings, dk_strings)
        w_eff = w.loc[s.index]

        freq = weighted_counts(s, w_eff) if use_weights else unweighted_counts(s)

        m1, m2, m3 = st.columns(3)
        m1.metric("N (rows used)", f"{len(s):,}")
        if use_weights:
            m2.metric("Sum of weights", f"{float(w_eff.sum()):,.2f}")
            m3.metric("Mean weight", f"{float(w_eff.mean()):,.4f}")
        else:
            m2.metric("Sum of weights", "—")
            m3.metric("Mean weight", "—")

        freq_show = freq.copy()
        if "count" in freq_show.columns:
            freq_show["count"] = pd.to_numeric(freq_show["count"], errors="coerce").round(0)
            freq_show["count"] = freq_show["count"].astype("Int64")
        if "percent" in freq_show.columns:
            freq_show["percent"] = pd.to_numeric(freq_show["percent"], errors="coerce").round(2)
        st.dataframe(freq_show, use_container_width=True)
        st.caption(f"N (rows used) = {len(s):,}" + (f" | Σταθμισμένο άθροισμα = {w_eff.sum():.2f}" if use_weights else ""))

    with colB:
        chart_type = st.selectbox("Τύπος διαγράμματος", SUPPORTED_CHARTS_UNIVAR, index=0, key="uni_chart")
        plot_univariate(freq, chart_type, title=f"{code}: {text}")


# -------- Tab 2 --------
with tab2:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        row_code, row_text, _ = select_variable_ui(catalog, key="ct_row", title="Ανεξάρτητη (Rows)")
        col_code, col_text, _ = select_variable_ui(catalog, key="ct_col", title="Εξαρτημένη (Columns)")
        st.caption(f"Rows: **{row_code}** — {row_text} | Cols: **{col_code}** — {col_text}")

        r = apply_missing_filters(df[row_code], drop_na99, drop_dk88, na_strings, dk_strings)
        c = apply_missing_filters(df[col_code], drop_na99, drop_dk88, na_strings, dk_strings)
        idx = r.index.intersection(c.index)
        r, c = r.loc[idx], c.loc[idx]
        w_eff = w.loc[idx]

        n_unw = len(idx)
        w_sum = float(w_eff.sum())
        w_mean = float(w_eff.mean()) if n_unw else float("nan")
        m1, m2, m3 = st.columns(3)
        m1.metric("N (unweighted)", f"{n_unw:,}")
        if use_weights:
            m2.metric("Sum of weights", f"{w_sum:,.2f}")
            m3.metric("Mean weight", f"{w_mean:,.4f}")
        else:
            m2.metric("Sum of weights", "—")
            m3.metric("Mean weight", "—")

        if use_weights:
            ct = pd.DataFrame({"r": r, "c": c, "w": w_eff}).pivot_table(index="r", columns="c", values="w", aggfunc="sum", fill_value=0.0)
            st.info("ℹ️ Weighted crosstab. Το χ²/p είναι προσεγγιστικό.")
        else:
            ct = pd.crosstab(r, c, dropna=False)

        ct_total = float(ct.values.sum())
        st.caption("Έλεγχος: άθροισμα κελιών πίνακα = " + (f"{ct_total:,.2f} (weighted)" if use_weights else f"{ct_total:,.0f} (N)"))

        st.subheader("Πίνακες")
        st.write("**(1) Απόλυτες τιμές**")
        st.dataframe(ct, use_container_width=True)

        st.write("**(2) Ποσοστά ανά γραμμή (100%)**")
        row_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
        st.dataframe(row_pct.round(2), use_container_width=True)

        st.write("**(3) Ποσοστά ανά στήλη (100%)**")
        col_pct = (ct.div(ct.sum(axis=0), axis=1) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
        st.dataframe(col_pct.round(2), use_container_width=True)

        st.subheader("Στατιστικά")
        try:
            chi2, p, dof, expected = stats.chi2_contingency(ct.values)
            v = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
            m1, m2, m3 = st.columns(3)
            m1.metric("Chi-square (χ²)", f"{chi2:.4f}")
            m2.metric("p-value", f"{p:.6g}")
            m3.metric("Cramer's V", f"{v:.4f}" if np.isfinite(v) else "—")

            min_exp = float(np.min(expected))
            st.caption(f"df = {dof} | min expected = {min_exp:.2f}")
            if min_exp < 5:
                st.warning("⚠️ Min expected < 5: το χ² μπορεί να μην είναι αξιόπιστο (σπάνιες κατηγορίες).")
        except Exception as e:
            st.error(f"Δεν μπόρεσα να υπολογίσω χ²: {e}")

        st.subheader("Standard deviation (μόνο αν η εξαρτημένη είναι αριθμητική)")
        y = pd.to_numeric(c, errors="coerce")
        if y.notna().sum() >= max(10, int(0.5*len(y))):
            if use_weights:
                ww = w_eff.loc[y.dropna().index]
                yy = y.dropna()
                mu = np.average(yy, weights=ww)
                var = np.average((yy - mu)**2, weights=ww)
                st.metric("Συνολικό weighted std", f"{math.sqrt(var):.4f}")
            else:
                st.metric("Συνολικό std", f"{y.std(ddof=1):.4f}")
        else:
            st.caption("Η εξαρτημένη μεταβλητή φαίνεται κατηγορική/λεκτική.")

    with right:
        st.subheader("Διαγράμματα")
        chart_type2 = st.selectbox("Τύπος διαγράμματος", SUPPORTED_CHARTS_BIVAR, index=0, key="ct_chart")
        plot_bivariate(ct, chart_type2, title=f"{row_code} × {col_code}")


# -------- Tab 3 --------
with tab3:
    st.subheader("PCA + Varimax (Κλίμακες στάσεων)")
    st.caption("Απαιτούνται αριθμητικά items (Likert). Αν το Excel έχει λεκτικά, χρειάζεται recoding.")
    cat_filter = st.selectbox("Φίλτρο κατηγορίας", ["Όλα"] + sorted(catalog["category"].unique()), key="pca_cat")
    subset = catalog if cat_filter == "Όλα" else catalog[catalog["category"] == cat_filter]
    options = (subset["text"] + "  [" + subset["code"] + "]").sort_values().tolist()
    chosen = st.multiselect("Διάλεξε μεταβλητές (items)", options, default=[], key="pca_vars")
    codes = [x.split("[")[-1].replace("]", "").strip() for x in chosen]

    if len(codes) < 2:
        st.info("Διάλεξε τουλάχιστον 2 items.")
    else:
        X_raw = df[codes].copy()
        for cc in codes:
            X_raw[cc] = apply_missing_filters(X_raw[cc], drop_na99, drop_dk88, na_strings, dk_strings)

        X_num = X_raw.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        st.caption(f"Rows used: {X_num.shape[0]:,} | Items: {X_num.shape[1]}")
        if X_num.shape[0] < 50:
            st.warning("Λίγες γραμμές μετά τον καθαρισμό.")

        scaler = StandardScaler()
        X = scaler.fit_transform(X_num.values)

        try:
            kmo, bart_chi2, bart_p = kmo_bartlett(X)
            a, b, c = st.columns(3)
            a.metric("KMO", f"{kmo:.3f}")
            b.metric("Bartlett χ²", f"{bart_chi2:.2f}")
            c.metric("Bartlett p", f"{bart_p:.6g}")
        except Exception as e:
            st.warning(f"KMO/Bartlett error: {e}")

        max_k = min(X.shape[1], 12)
        pca0 = PCA(n_components=min(X.shape[1], X.shape[0]-1)).fit(X)
        eigvals = pca0.explained_variance_
        kaiser = int((eigvals > 1.0).sum())
        k_default = max(1, min(max_k, kaiser if kaiser > 0 else 2))
        k = st.slider("Αριθμός παραγόντων", 1, max_k, k_default)

        pca = PCA(n_components=k).fit(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        rot_loadings, _R = varimax(loadings)

        evr = pca.explained_variance_ratio_
        expl = pd.DataFrame({
            "Factor": [f"F{i+1}" for i in range(k)],
            "Explained %": (evr*100).round(2),
            "Cumulative %": (np.cumsum(evr)*100).round(2),
            "Eigenvalue": pca.explained_variance_.round(3),
        })
        st.subheader("Explained variance")
        st.dataframe(expl, use_container_width=True)

        load_tbl = pd.DataFrame(rot_loadings, index=codes, columns=[f"F{i+1}" for i in range(k)])
        st.subheader("Rotated loadings (varimax)")
        st.dataframe(load_tbl.round(3), use_container_width=True)

        st.subheader("Αξιοπιστία")
        alpha_all = cronbach_alpha(X_num.values)
        st.metric("Cronbach α (όλα τα items)", f"{alpha_all:.3f}" if np.isfinite(alpha_all) else "—")

        thr = st.slider("Threshold |loading|", 0.2, 0.8, 0.4, 0.05)
        rel_rows = []
        for j in range(k):
            items = load_tbl.index[load_tbl.iloc[:, j].abs() >= thr].tolist()
            a = cronbach_alpha(X_num[items].values) if len(items) >= 2 else np.nan
            rel_rows.append({"Factor": f"F{j+1}", "n_items": len(items), "alpha": a, "items": ", ".join(items)})
        st.dataframe(pd.DataFrame(rel_rows), use_container_width=True)

        st.subheader("Factor scores + download")
        scores_rot = X @ rot_loadings
        scores_df = pd.DataFrame(scores_rot, index=X_num.index, columns=[f"F{i+1}_score" for i in range(k)])
        st.dataframe(scores_df.head(10).round(3), use_container_width=True)

        out_df = df.copy()
        for c in scores_df.columns:
            out_df.loc[scores_df.index, c] = scores_df[c]

        st.download_button(
            "⬇️ Download CSV με factor scores",
            data=out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="data_with_factor_scores.csv",
            mime="text/csv",
        )

        st.subheader("Regression (OLS)")
        if not HAS_STATSMODELS:
            st.info("statsmodels δεν είναι διαθέσιμο (βάλε το στο requirements).")
        else:
            dep_sel = st.selectbox("Dependent (αριθμητική)", options, key="reg_y")
            y_code = dep_sel.split("[")[-1].replace("]", "").strip()
            predictors = st.multiselect("Predictors", list(out_df.columns), default=list(scores_df.columns), key="reg_x")

            yv = pd.to_numeric(out_df[y_code], errors="coerce")
            Xv = out_df[predictors].apply(pd.to_numeric, errors="coerce")
            reg_df = pd.concat([yv.rename("y"), Xv], axis=1).dropna(axis=0, how="any")

            if len(reg_df) < 50:
                st.warning("Λίγες γραμμές για regression μετά τον καθαρισμό.")
            else:
                Xmat = sm.add_constant(reg_df.drop(columns=["y"]))
                model = sm.OLS(reg_df["y"], Xmat).fit()
                st.text(model.summary().as_text())
