"""
AMS Susceptibility Prediction — Streamlit App
Logistic Regression deployment | Baseline Features
Theme-aware: supports both light and dark mode + manual toggle
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMS Risk Predictor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True   # default to dark

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

is_dark = st.session_state.dark_mode

# ── Theme color palettes ──────────────────────────────────────────────────────
if is_dark:
    # ── Dark theme colours ────────────────────────────────────────────────────
    BG_PAGE        = "#0d0d1a"
    BG_SIDEBAR     = "#0f0f1a"
    BG_CARD        = "#1e1e2e"
    BG_INFO        = "#1a3a5c"
    BG_WARN        = "#3a2e00"
    BG_SECTION     = "#0d0d1a"

    TEXT_PRIMARY   = "#FFFFFF"
    TEXT_SECONDARY = "#E0E0E0"
    TEXT_MUTED     = "#B0C4DE"
    TEXT_INFO      = "#B3E5FC"
    TEXT_WARN      = "#FFE082"
    TEXT_CODE      = "#FFD54F"

    ACCENT_BLUE    = "#7EB8F7"
    ACCENT_CYAN    = "#4FC3F7"
    BORDER_INFO    = "#4FC3F7"
    BORDER_WARN    = "#FFD54F"
    BORDER_CARD    = "#7EB8F7"

    RISK_HIGH_BG   = "linear-gradient(135deg, #ff4b4b22, #ff4b4b44)"
    RISK_HIGH_BOR  = "#FF6B6B"
    RISK_LOW_BG    = "linear-gradient(135deg, #4FC3F722, #4FC3F744)"
    RISK_LOW_BOR   = "#4FC3F7"

    TAB_COLOR      = "#B0C4DE"
    TAB_ACTIVE     = "#7EB8F7"

    TH_BG          = "#1a1a2e"
    TH_COLOR       = "#7EB8F7"
    TD_COLOR       = "#E0E0E0"
    CODE_BG        = "#2a2a3e"

    TOGGLE_LABEL   = "☀️ Switch to Light Mode"
    MPL_STYLE      = "dark_background"
    MPL_TEXT       = "white"
    MPL_GRID       = "#444"
    MPL_SPINE      = "#555"
else:
    # ── Light theme colours ───────────────────────────────────────────────────
    BG_PAGE        = "#f5f7fa"
    BG_SIDEBAR     = "#eef1f7"
    BG_CARD        = "#ffffff"
    BG_INFO        = "#e3f2fd"
    BG_WARN        = "#fff8e1"
    BG_SECTION     = "#f5f7fa"

    TEXT_PRIMARY   = "#1a1a2e"
    TEXT_SECONDARY = "#2c3e50"
    TEXT_MUTED     = "#546e7a"
    TEXT_INFO      = "#0d47a1"
    TEXT_WARN      = "#e65100"
    TEXT_CODE      = "#bf360c"

    ACCENT_BLUE    = "#1565c0"
    ACCENT_CYAN    = "#0277bd"
    BORDER_INFO    = "#0277bd"
    BORDER_WARN    = "#f9a825"
    BORDER_CARD    = "#1565c0"

    RISK_HIGH_BG   = "linear-gradient(135deg, #ffebee, #ffcdd2)"
    RISK_HIGH_BOR  = "#e53935"
    RISK_LOW_BG    = "linear-gradient(135deg, #e3f2fd, #bbdefb)"
    RISK_LOW_BOR   = "#1565c0"

    TAB_COLOR      = "#546e7a"
    TAB_ACTIVE     = "#1565c0"

    TH_BG          = "#e8edf5"
    TH_COLOR       = "#1565c0"
    TD_COLOR       = "#2c3e50"
    CODE_BG        = "#f0f4f8"

    TOGGLE_LABEL   = "🌙 Switch to Dark Mode"
    MPL_STYLE      = "seaborn-v0_8-whitegrid"
    MPL_TEXT       = "#1a1a2e"
    MPL_GRID       = "#ccc"
    MPL_SPINE      = "#aaa"


# ── Dynamic CSS injection ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* ── Page background ── */
    .stApp {{
        background-color: {BG_PAGE} !important;
    }}
    .main .block-container {{
        background-color: {BG_PAGE} !important;
    }}

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR} !important;
    }}
    div[data-testid="stSidebar"] * {{
        color: {TEXT_SECONDARY} !important;
    }}
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {{
        color: {ACCENT_BLUE} !important;
    }}
    div[data-testid="stSidebar"] table th {{
        color: {ACCENT_BLUE} !important;
        background-color: {TH_BG} !important;
    }}
    div[data-testid="stSidebar"] table td {{
        color: {TD_COLOR} !important;
    }}

    /* ── Main headings ── */
    .main-header {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        font-size: 1.0rem;
        color: {TEXT_MUTED};
        margin-bottom: 1.5rem;
    }}
    h1, h2 {{
        color: {ACCENT_BLUE} !important;
        font-weight: 800 !important;
    }}
    h3 {{
        color: {ACCENT_BLUE} !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }}

    /* ── Risk boxes ── */
    .risk-high {{
        background: {RISK_HIGH_BG};
        border: 2px solid {RISK_HIGH_BOR};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }}
    .risk-low {{
        background: {RISK_LOW_BG};
        border: 2px solid {RISK_LOW_BOR};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }}

    /* ── Metric card ── */
    .metric-card {{
        background: {BG_CARD};
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid {BORDER_CARD};
        margin-bottom: 0.5rem;
    }}

    /* ── Section header ── */
    .section-header {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {ACCENT_BLUE};
        border-bottom: 3px solid {ACCENT_BLUE};
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        letter-spacing: 0.01em;
    }}

    /* ── Info / warning boxes ── */
    .info-box {{
        background: {BG_INFO};
        border-left: 4px solid {BORDER_INFO};
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: {TEXT_INFO};
    }}
    .warning-box {{
        background: {BG_WARN};
        border-left: 4px solid {BORDER_WARN};
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: {TEXT_WARN};
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.95rem;
        font-weight: 500;
        color: {TAB_COLOR} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TAB_ACTIVE} !important;
        border-bottom: 3px solid {TAB_ACTIVE} !important;
    }}

    /* ── Body text ── */
    p, li {{
        color: {TEXT_SECONDARY} !important;
    }}
    .stMarkdown p {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ── Tables ── */
    thead tr th {{
        color: {TH_COLOR} !important;
        font-weight: 700 !important;
        background-color: {TH_BG} !important;
    }}
    tbody tr td {{
        color: {TD_COLOR} !important;
    }}

    /* ── Code ── */
    code {{
        color: {TEXT_CODE} !important;
        background-color: {CODE_BG} !important;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
    }}

    /* ── Streamlit metric labels ── */
    .stMetric label {{
        color: {TEXT_MUTED} !important;
        font-weight: 600 !important;
    }}
    .stMetric [data-testid="metric-container"] {{
        background-color: {BG_CARD};
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        border: 1px solid {BORDER_CARD}44;
    }}

    /* ── Dataframes ── */
    .stDataFrame {{
        background-color: {BG_CARD} !important;
    }}

    /* ── Input widgets ── */
    .stFileUploader label,
    .stSelectbox label,
    .stTextInput label {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ── Theme toggle button ── */
    .theme-toggle-btn button {{
        background: {BG_CARD} !important;
        color: {TEXT_PRIMARY} !important;
        border: 2px solid {ACCENT_BLUE} !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        padding: 0.3rem 1rem !important;
        transition: all 0.2s ease !important;
    }}
    .theme-toggle-btn button:hover {{
        background: {ACCENT_BLUE} !important;
        color: {BG_PAGE} !important;
    }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        color: {TEXT_SECONDARY} !important;
        background-color: {BG_CARD} !important;
    }}
    .streamlit-expanderContent {{
        background-color: {BG_CARD} !important;
    }}

    /* ── Alert / info boxes from st.info / st.error ── */
    div[data-testid="stNotificationContent"] p {{
        color: inherit !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib theme helper ───────────────────────────────────────────────────
def apply_mpl_theme(fig, axes_list=None):
    """Apply consistent matplotlib colours matching the current UI theme."""
    fig.patch.set_facecolor(BG_PAGE if not is_dark else "#0d0d1a")
    if axes_list is None:
        axes_list = fig.get_axes()
    for ax in axes_list:
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=MPL_TEXT)
        ax.xaxis.label.set_color(MPL_TEXT)
        ax.yaxis.label.set_color(MPL_TEXT)
        ax.title.set_color(MPL_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_SPINE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(MPL_TEXT)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_model_bundle():
    """Try to load the serialised model; return None if not found."""
    search_paths = [
        Path("models/final_model_LR.pkl"),
        Path("app/models/final_model_LR.pkl"),
        Path("../models/final_model_LR.pkl"),
    ]
    for p in search_paths:
        if p.exists():
            return load_model(str(p))
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mountain.png", width=72)
    st.markdown(f"## 🏔️ AMS Risk Predictor")
    st.markdown("**Logistic Regression** | Baseline Features")
    st.markdown("---")
    st.markdown(f"""
**Model Performance (Nested LOOCV, n=21)**

| Metric | Value |
|---|---|
| AUC-ROC | **0.868** |
| Sensitivity | **1.000** |
| Specificity | **0.750** |
| F1 Score | **0.971** |
| MCC | **0.842** |
| False Negatives | **0** |
""")
    st.markdown("---")
    st.markdown(f"""
<div class="warning-box">
⚠️ <b>Research use only.</b> Validate with an independent cohort before clinical deployment.
</div>
""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"""
<div class="info-box">
ℹ️ AMS = Acute Mountain Sickness. Model uses pre-exposure baseline gene expression + physiology.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    # ── Theme toggle in sidebar ───────────────────────────────────────────────
    st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    st.button(TOGGLE_LABEL, on_click=toggle_theme, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f"<small style='color:{TEXT_MUTED};'>Currently: {'🌙 Dark Mode' if is_dark else '☀️ Light Mode'}</small>",
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
header_col, toggle_col = st.columns([5, 1])
with header_col:
    st.markdown(f'<div class="main-header">🏔️ AMS Susceptibility Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Logistic Regression · Baseline Transcriptomic + Physiological Features · Nested LOOCV Validated</div>', unsafe_allow_html=True)
with toggle_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    st.button(
        "☀️ Light" if is_dark else "🌙 Dark",
        on_click=toggle_theme,
        key="header_toggle",
        help="Toggle light/dark theme",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Batch Predict (CSV)", "📈 Model Info"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Batch prediction via CSV / Excel
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Batch Prediction — CSV or Excel Upload</div>', unsafe_allow_html=True)
    st.markdown(f"""
Upload a **CSV or Excel file** where:
- **Column 1** → `subject_id` (non-negative integer, unique per row)
- **Remaining columns** → feature values (one column per model feature)
- **Each row** → one subject

Subjects with **any missing cell** will be **flagged and skipped** — predictions are only made for complete records.
""")

    bundle1 = get_model_bundle()

    if bundle1 is None:
        st.markdown(f"""
<div class="info-box">
No model found in <code>models/final_model_LR.pkl</code>. Upload it below.
</div>
""", unsafe_allow_html=True)
        uploaded_model1 = st.file_uploader("Upload `final_model_LR.pkl`", type=["pkl"], key="model1")
        if uploaded_model1:
            bundle1 = pickle.load(uploaded_model1)

    if bundle1:
        model1      = bundle1["model"]
        imputer1    = bundle1["imputer"]
        scaler1     = bundle1["scaler"]
        feat_names1 = bundle1["feat_names"]

        # ── Step 1: Template download ──────────────────────────────────────
        st.markdown("### Step 1 — Download the Input Template")
        st.markdown("Fill in this template and upload it in Step 2. Column order must be: `subject_id` then all feature columns.")

        template_df = pd.DataFrame(columns=["subject_id"] + feat_names1)
        example_rows = pd.DataFrame(
            [[i] + [None] * len(feat_names1) for i in range(1, 4)],
            columns=["subject_id"] + feat_names1,
        )
        template_df = pd.concat([template_df, example_rows], ignore_index=True)

        col_dl1, col_dl2 = st.columns(2)
        csv_template = template_df.to_csv(index=False)
        col_dl1.download_button(
            "⬇️ Download CSV Template",
            data=csv_template,
            file_name="ams_input_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        try:
            import openpyxl
            from io import BytesIO
            excel_buf = BytesIO()
            template_df.to_excel(excel_buf, index=False, engine="openpyxl")
            col_dl2.download_button(
                "⬇️ Download Excel Template",
                data=excel_buf.getvalue(),
                file_name="ams_input_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            col_dl2.info("Install `openpyxl` for Excel template download.")

        st.markdown(f"Template has **1 + {len(feat_names1)} columns** (subject_id + {len(feat_names1)} features).")

        st.markdown("---")

        # ── Step 2: Upload file ────────────────────────────────────────────
        st.markdown("### Step 2 — Upload Your Filled File")
        batch_file = st.file_uploader(
            "Upload CSV or Excel (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            key="batch",
            help="First column must be subject_id (non-negative integer). Remaining columns = model features.",
        )

        if batch_file:
            try:
                fname = batch_file.name.lower()
                if fname.endswith(".csv"):
                    raw_df = pd.read_csv(batch_file)
                else:
                    raw_df = pd.read_excel(batch_file, engine="openpyxl")

                st.markdown(f"**Loaded:** {len(raw_df)} rows × {len(raw_df.columns)} columns")

                if "subject_id" not in raw_df.columns:
                    st.error("❌ Column `subject_id` not found. First column must be named `subject_id`.")
                    st.stop()

                bad_ids = []
                for idx, val in enumerate(raw_df["subject_id"]):
                    try:
                        v = float(val)
                        if v < 0 or v != int(v):
                            bad_ids.append((idx + 1, val))
                    except (ValueError, TypeError):
                        bad_ids.append((idx + 1, val))

                if bad_ids:
                    st.error(f"❌ `subject_id` must be a non-negative integer. Problems at rows: {bad_ids[:5]}")
                    st.stop()

                raw_df["subject_id"] = raw_df["subject_id"].astype(int)

                dupes = raw_df["subject_id"][raw_df["subject_id"].duplicated()].tolist()
                if dupes:
                    st.warning(f"⚠️ Duplicate subject_id values found: {dupes}. Each row should be a unique subject.")

                cols_in_file         = [c for c in raw_df.columns if c != "subject_id"]
                missing_feature_cols = [f for f in feat_names1 if f not in raw_df.columns]
                extra_cols           = [c for c in cols_in_file if c not in feat_names1]

                if missing_feature_cols:
                    st.error(
                        f"❌ **{len(missing_feature_cols)} required feature column(s) are missing from the file.** "
                        f"These columns must be present even if values are empty:\n\n"
                        + ", ".join(f"`{c}`" for c in missing_feature_cols[:10])
                        + ("..." if len(missing_feature_cols) > 10 else "")
                    )
                    st.markdown("Download the template above and make sure all feature columns are included.")
                    st.stop()

                if extra_cols:
                    st.info(f"ℹ️ {len(extra_cols)} extra column(s) in file will be ignored: {extra_cols[:5]}{'...' if len(extra_cols)>5 else ''}")

                feat_df = raw_df[feat_names1].copy()

                for col in feat_names1:
                    feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

                missing_per_subject = feat_df.isnull().sum(axis=1)
                missing_per_col     = feat_df.isnull().sum(axis=0)

                complete_mask   = missing_per_subject == 0
                incomplete_mask = ~complete_mask

                n_total     = len(raw_df)
                n_complete  = complete_mask.sum()
                n_incomplete = incomplete_mask.sum()

                st.markdown("---")
                st.markdown("### Data Quality Report")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Subjects", n_total)
                kpi2.metric("✅ Complete (will predict)", n_complete)
                kpi3.metric("⚠️ Incomplete (flagged)", n_incomplete,
                            delta=f"-{n_incomplete}" if n_incomplete > 0 else None,
                            delta_color="inverse")
                kpi4.metric("Features Required", len(feat_names1))

                if n_incomplete > 0:
                    st.markdown("#### ⚠️ Flagged Subjects — Missing Data")
                    flagged_df = raw_df[incomplete_mask][["subject_id"]].copy()
                    flagged_df["Missing_Count"] = missing_per_subject[incomplete_mask].values
                    flagged_df["Missing_Features"] = [
                        ", ".join(feat_names1[j] for j in range(len(feat_names1))
                                  if pd.isnull(feat_df.iloc[i, j]))
                        for i in raw_df.index[incomplete_mask]
                    ]
                    flagged_df = flagged_df.reset_index(drop=True)

                    st.dataframe(
                        flagged_df.style.map(
                            lambda v: "background-color: #fff3cd; color: #856404;",
                            subset=["Missing_Count"],
                        ),
                        use_container_width=True,
                    )

                    cols_with_missing = missing_per_col[missing_per_col > 0].sort_values(ascending=False)
                    if len(cols_with_missing) > 0:
                        with st.expander(f"📋 Features with missing values ({len(cols_with_missing)} features)"):
                            miss_summary = pd.DataFrame({
                                "Feature": cols_with_missing.index,
                                "Missing Count": cols_with_missing.values,
                                "Affected Subjects": [
                                    ", ".join(str(s) for s in
                                              raw_df.loc[feat_df[col].isnull(), "subject_id"].tolist())
                                    for col in cols_with_missing.index
                                ],
                            })
                            st.dataframe(miss_summary, use_container_width=True)

                if n_complete == 0:
                    st.error("❌ No subjects with complete data. Please fix missing values and re-upload.")
                else:
                    st.markdown("---")
                    st.markdown(f"### 🔮 Predictions — {n_complete} Complete Subject(s)")

                    complete_df = raw_df[complete_mask].copy()
                    X_complete  = feat_df[complete_mask].values.astype(float)

                    X_sc = scaler1.transform(X_complete)

                    probs = model1.predict_proba(X_sc)[:, 1]
                    preds = model1.predict(X_sc)

                    def risk_level(p):
                        if p < 0.30:   return "Low"
                        elif p < 0.50: return "Moderate"
                        elif p < 0.70: return "High"
                        else:          return "Very High"

                    results_df = pd.DataFrame({
                        "subject_id":     complete_df["subject_id"].values,
                        "AMS_Prediction": ["AMS+" if p == 1 else "AMS-" for p in preds],
                        "P(AMS+)":        probs.round(4),
                        "P(AMS-)":        (1 - probs).round(4),
                        "Risk_Level":     [risk_level(p) for p in probs],
                        "Status":         ["Predicted"] * n_complete,
                    })

                    def style_prediction(val):
                        if val == "AMS+":
                            return "color: #D62728; font-weight: bold"
                        elif val == "AMS-":
                            return "color: #1F77B4; font-weight: bold"
                        return ""

                    st.dataframe(
                        results_df.style.map(style_prediction, subset=["AMS_Prediction"]),
                        use_container_width=True,
                        height=int(min(400, 60 + 35 * n_complete)),
                    )

                    n_pos = (preds == 1).sum()
                    n_neg = (preds == 0).sum()
                    mean_prob = probs.mean()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("AMS+ Predicted", n_pos,
                              help="Subjects predicted susceptible to AMS")
                    m2.metric("AMS− Predicted", n_neg,
                              help="Subjects predicted resistant to AMS")
                    m3.metric("Mean P(AMS+)", f"{mean_prob:.3f}",
                              help="Average AMS+ probability across predicted subjects")

                    # ── Charts — theme-aware ───────────────────────────────
                    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4),
                                               facecolor=BG_PAGE if not is_dark else "#0d0d1a")
                    apply_mpl_theme(fig3, list(axes3))

                    ax = axes3[0]
                    ax.set_facecolor(BG_CARD)
                    labels_plot, counts_plot, colors_plot = [], [], []
                    if n_pos > 0:
                        labels_plot.append("AMS+"); counts_plot.append(n_pos); colors_plot.append("#D62728")
                    if n_neg > 0:
                        labels_plot.append("AMS-"); counts_plot.append(n_neg); colors_plot.append("#1F77B4")
                    bars = ax.bar(labels_plot, counts_plot, color=colors_plot, alpha=0.85,
                                  edgecolor="white", width=0.5)
                    for bar, cnt in zip(bars, counts_plot):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                                str(cnt), ha="center", va="bottom",
                                fontweight="bold", fontsize=12, color=MPL_TEXT)
                    ax.set_title("Prediction Distribution", fontweight="bold", fontsize=11, color=MPL_TEXT)
                    ax.set_ylabel("Number of Subjects", color=MPL_TEXT)
                    ax.set_ylim(0, max(counts_plot) * 1.25)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    ax = axes3[1]
                    ax.set_facecolor(BG_CARD)
                    subj_labels = [str(s) for s in results_df["subject_id"].values]
                    bar_colors  = ["#D62728" if p == 1 else "#1F77B4" for p in preds]
                    ax.bar(subj_labels, probs, color=bar_colors, alpha=0.82, edgecolor="white")
                    ax.axhline(0.5, color=MPL_TEXT, lw=1.2, ls="--", alpha=0.6, label="Threshold (0.5)")
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Subject ID", color=MPL_TEXT)
                    ax.set_ylabel("P(AMS+)", color=MPL_TEXT)
                    ax.set_title("AMS+ Probability per Subject", fontweight="bold", fontsize=11, color=MPL_TEXT)
                    ax.legend(fontsize=8, labelcolor=MPL_TEXT,
                              facecolor=BG_CARD, edgecolor=MPL_SPINE)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, color=MPL_TEXT)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    ax = axes3[2]
                    ax.set_facecolor(BG_CARD)
                    if n_complete > 1:
                        ax.hist(probs, bins=min(8, n_complete), color="#4C72B0",
                                alpha=0.75, edgecolor="white")
                    else:
                        ax.bar([probs[0]], [1], color="#4C72B0", alpha=0.75, width=0.05)
                    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Threshold")
                    ax.set_xlabel("P(AMS+)", color=MPL_TEXT)
                    ax.set_ylabel("Count", color=MPL_TEXT)
                    ax.set_title("Probability Distribution", fontweight="bold", fontsize=11, color=MPL_TEXT)
                    ax.legend(fontsize=8, labelcolor=MPL_TEXT,
                              facecolor=BG_CARD, edgecolor=MPL_SPINE)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close()

                    st.markdown("### 📋 Full Subject Report (All Subjects)")

                    all_report = raw_df[["subject_id"]].copy()
                    pred_map   = dict(zip(complete_df["subject_id"].values,
                                         zip(["AMS+" if p == 1 else "AMS-" for p in preds],
                                             probs.round(4),
                                             [risk_level(p) for p in probs])))

                    all_report["AMS_Prediction"] = all_report["subject_id"].map(
                        lambda s: pred_map[s][0] if s in pred_map else "—"
                    )
                    all_report["P(AMS+)"] = all_report["subject_id"].map(
                        lambda s: pred_map[s][1] if s in pred_map else np.nan
                    )
                    all_report["Risk_Level"] = all_report["subject_id"].map(
                        lambda s: pred_map[s][2] if s in pred_map else "—"
                    )
                    all_report["Missing_Values"] = missing_per_subject.values
                    all_report["Status"] = all_report["subject_id"].map(
                        lambda s: "Predicted" if s in pred_map else "Skipped !! (incomplete data)"
                    )

                    def style_status(val):
                        if "Skipped" in str(val):
                            return "background-color: #fff3cd; color: #856404; font-weight: bold"
                        elif "Predicted" in str(val):
                            return "background-color: #d4edda; color: #155724; font-weight: bold"
                        return ""

                    st.dataframe(
                        all_report.style
                            .map(style_prediction, subset=["AMS_Prediction"])
                            .map(style_status, subset=["Status"]),
                        use_container_width=True,
                    )

                    st.markdown("### ⬇️ Download Results")
                    dl1, dl2 = st.columns(2)

                    pred_csv = results_df.to_csv(index=False)
                    dl1.download_button(
                        "⬇️ Download Predictions Only (CSV)",
                        data=pred_csv,
                        file_name="ams_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    full_csv = all_report.to_csv(index=False)
                    dl2.download_button(
                        "⬇️ Download Full Report (CSV)",
                        data=full_csv,
                        file_name="ams_full_report.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.exception(e)

    else:
        st.info("📂 Please upload the model file above.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Model information
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📈 Model Architecture & Validation</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Pipeline Summary")
        st.markdown("""
| Stage | Detail |
|---|---|
| **Model** | Logistic Regression |
| **Solver** | SAGA |
| **Regularisation** | L1 / L2 (selected via inner CV) |
| **Imputation** | Median (RobustScaler) |
| **Scaling** | RobustScaler |
| **Evaluation** | Nested LOOCV (outer: LOO, inner: StratifiedKFold-3) |
| **Feature matrix** | BL_full — n=21 subjects |
| **Feature count** | 67 (60 genes + 6 physiology + sex) |
| **Class imbalance** | AMS+: 17 / AMS-: 4  |
""")

    with col_b:
        st.markdown("### LOOCV Performance")
        st.markdown("""
| Metric | Value | Clinical Weight |
|---|---|---|
| **AUC-ROC** | 0.868 | High |
| **Sensitivity** | **1.000** | **Critical** |
| **Specificity** | 0.750 | Moderate |
| **F1 Score** | 0.971 | High |
| **MCC** | 0.842 | High |
| **FN (missed AMS+)** | **0** | **Critical** |
| **FP (false alarms)** | 1 | Moderate |
| Precision | 0.944 | Moderate |
| Accuracy | 0.952 | Low |
""")

    st.markdown('<div class="section-header">⚖️ Why Logistic Regression over Random Forest?</div>', unsafe_allow_html=True)
    st.markdown("""
| Metric | RF | LR | Winner |
|---|---|---|---|
| AUC-ROC | 0.956 | 0.868 | RF |
| Sensitivity | 1.000 | 1.000 | Tie |
| Specificity | 0.500 | **0.750** | **LR** |
| F1 Score | 0.944 | **0.971** | **LR** |
| MCC | 0.669 | **0.842** | **LR** |
| FN (missed AMS+) | 0 | 0 | Tie |
| FP (false alarms) | 2 | **1** | **LR** |
| Interpretability | SHAP needed | **Direct** | **LR** |
| Overfitting risk | Higher | **Lower** | **LR** |

**LR is recommended for deployment** — fewer false alarms, higher MCC, lower overfitting risk at n=21, and auditable coefficients. RF is the better tool for **biological discovery** via SHAP.
""")

    st.markdown('<div class="section-header">🧬 Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown("""
Features were selected by **convergence across 4 evidence streams** (Notebook 2 → 3):

- **S1** — Baseline DE composite score (top-50 genes)
- **S2** — Baseline DE nominal p < 0.05
- **S3** — Delta DE composite score (top-50 genes)
- **S4** — Spearman |ρ| ≥ 0.40 (delta gene × delta physiology, AMS+ stratum)

Genes appearing in **≥ 2 streams** were selected, yielding a compact convergent feature set that balances statistical evidence with biological plausibility.

Additional engineered features:
- **Hypoxic Response Score (HRS)** — mean expression of hypoxia-responsive genes
- **Physiological Stress Composite (PSC)** — standardised physiology index
- **PCA components** — capturing residual transcriptomic variance
""")

    bundle2 = get_model_bundle()
    if bundle2 and hasattr(bundle2["model"], "coef_"):
        st.markdown('<div class="section-header">📊 Model Coefficients (Global Feature Importance)</div>', unsafe_allow_html=True)
        model2      = bundle2["model"]
        feat_names2 = bundle2["feat_names"]
        coefs       = model2.coef_[0]
        coef_df     = pd.DataFrame({
            "Feature": feat_names2,
            "Coefficient": coefs,
            "|Coefficient|": np.abs(coefs),
        }).sort_values("|Coefficient|", ascending=False).head(20)

        fig4, ax4 = plt.subplots(figsize=(9, 6),
                                  facecolor=BG_PAGE if not is_dark else "#0d0d1a")
        apply_mpl_theme(fig4, [ax4])
        ax4.set_facecolor(BG_CARD)

        colors4 = ["#D62728" if v > 0 else "#1F77B4" for v in coef_df["Coefficient"].values[::-1]]
        ax4.barh(range(len(coef_df)), coef_df["Coefficient"].values[::-1],
                 color=colors4, alpha=0.82, edgecolor="white")
        ax4.set_yticks(range(len(coef_df)))
        ax4.set_yticklabels([f[:30] for f in coef_df["Feature"].values[::-1]],
                            fontsize=8, color=MPL_TEXT)
        ax4.axvline(0, color=MPL_SPINE, lw=0.8)
        ax4.set_xlabel("Coefficient (positive = increases AMS+ risk)",
                       fontsize=9, color=MPL_TEXT)
        ax4.set_title("Top 20 LR Feature Coefficients\n(red = AMS+ risk ↑ | blue = AMS+ risk ↓)",
                      fontsize=10, fontweight="bold", color=MPL_TEXT)
        patches = [
            mpatches.Patch(color="#D62728", label="Increases AMS+ risk"),
            mpatches.Patch(color="#1F77B4", label="Decreases AMS+ risk"),
        ]
        ax4.legend(handles=patches, fontsize=8, labelcolor=MPL_TEXT,
                   facecolor=BG_CARD, edgecolor=MPL_SPINE)
        for spine in ["top", "right"]:
            ax4.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.dataframe(coef_df.reset_index(drop=True).round(4), use_container_width=True)
