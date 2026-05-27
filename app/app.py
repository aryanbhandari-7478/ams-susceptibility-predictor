"""
AMS Susceptibility Prediction — Streamlit App
Logistic Regression deployment | Baseline Features
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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #D62728;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #1F77B422, #1F77B444);
        border: 2px solid #1F77B4;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #4C72B0;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1565C0;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #FFC107;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #795548;
    }
    div[data-testid="stSidebar"] {
        background: #f0f2f6;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## 🏔️ AMS Risk Predictor")
    st.markdown("**Logistic Regression** | Baseline Features")
    st.markdown("---")
    st.markdown("""
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
    st.markdown("""
<div class="warning-box">
⚠️ <b>Research use only.</b> Validate with an independent cohort before clinical deployment.
</div>
""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
<div class="info-box">
ℹ️ AMS = Acute Mountain Sickness. Model uses pre-exposure baseline gene expression + physiology.
</div>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏔️ AMS Susceptibility Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Logistic Regression · Baseline Transcriptomic + Physiological Features · Nested LOOCV Validated</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Batch Predict (CSV)", "📈 Model Info", "📋 About"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Single subject prediction
# ══════════════════════════════════════════════════════════════════════
with tab1:
    bundle = get_model_bundle()

    if bundle is None:
        st.markdown('<div class="section-header">Upload Your Trained Model</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-box">
No model file found in <code>models/final_model_LR.pkl</code>.<br>
Please upload the serialised model from your <code>ams_pred/models/</code> folder.
</div>
""", unsafe_allow_html=True)
        st.markdown("")
        uploaded_model = st.file_uploader(
            "Upload `final_model_LR.pkl`",
            type=["pkl"],
            help="Generated by Notebook 4 → Section 16 → models/final_model_LR.pkl",
        )
        if uploaded_model:
            bundle = pickle.load(uploaded_model)
            st.success("✅ Model loaded successfully!")

    if bundle:
        model      = bundle["model"]
        imputer    = bundle["imputer"]
        scaler     = bundle["scaler"]
        feat_names = bundle["feat_names"]
        lr_metrics = bundle.get("loocv_metrics", {})
        modal_p    = bundle.get("modal_params", {})

        # Split features into physiological and gene groups
        PHYSIO_FEATS = [
            "PaO2 (mmHg)", "PaCO2 (mmHg)", "SaO2 (%)",
            "CaO2 (ml/dL)", "P50 (mmHg)", "Hemoglobin (g/dL)",
        ]
        physio_in_model = [f for f in PHYSIO_FEATS if f in feat_names]
        sex_in_model    = "sex_encoded" in feat_names
        gene_feats      = [f for f in feat_names
                           if f not in PHYSIO_FEATS and f != "sex_encoded"
                           and not f.startswith("bl_pca") and not f.startswith("bl_HRS")
                           and not f.startswith("bl_PSC") and not f.startswith("hypoxic")]
        engineered      = [f for f in feat_names if f not in physio_in_model
                           and f != "sex_encoded" and f not in gene_feats]

        st.markdown('<div class="section-header">📝 Subject Input</div>', unsafe_allow_html=True)
        st.markdown("Enter baseline (sea-level) measurements for the subject. Gene expression values are in log-normalised counts.")

        input_vals = {}

        # ── Physiological inputs ───────────────────────────────────────────
        if physio_in_model or sex_in_model:
            st.markdown("**Physiological Parameters**")
            physio_defaults = {
                "PaO2 (mmHg)": 90.0, "PaCO2 (mmHg)": 38.0, "SaO2 (%)": 97.0,
                "CaO2 (ml/dL)": 20.0, "P50 (mmHg)": 27.0, "Hemoglobin (g/dL)": 14.5,
            }
            physio_ranges = {
                "PaO2 (mmHg)":     (50.0,  130.0),
                "PaCO2 (mmHg)":    (20.0,   60.0),
                "SaO2 (%)":        (80.0,  100.0),
                "CaO2 (ml/dL)":    ( 8.0,   30.0),
                "P50 (mmHg)":      (20.0,   40.0),
                "Hemoglobin (g/dL)":(8.0,   20.0),
            }
            cols = st.columns(3)
            for idx, feat in enumerate(physio_in_model):
                lo, hi = physio_ranges.get(feat, (0.0, 200.0))
                input_vals[feat] = cols[idx % 3].number_input(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=float(physio_defaults.get(feat, (lo + hi) / 2)),
                    step=0.1, format="%.2f",
                )
            if sex_in_model:
                sex_choice = cols[len(physio_in_model) % 3].selectbox("Sex", ["Male", "Female"])
                input_vals["sex_encoded"] = 1 if sex_choice == "Male" else 0

        # ── Gene expression inputs ─────────────────────────────────────────
        if gene_feats:
            with st.expander(f"🧬 Gene Expression Values ({len(gene_feats)} genes)", expanded=False):
                st.markdown("""
<div class="info-box">
Enter baseline log-normalised expression values for each gene (ENSG IDs).  
Typical range: 0–20. Leave at 0 if unknown — the model will impute with training medians.
</div>
""", unsafe_allow_html=True)
                st.markdown("")
                gene_cols = st.columns(4)
                for idx, gene in enumerate(gene_feats):
                    input_vals[gene] = gene_cols[idx % 4].number_input(
                        gene, min_value=0.0, max_value=50.0, value=0.0,
                        step=0.01, format="%.4f",
                        help=f"Baseline log-normalised expression for {gene}",
                    )

        # ── Engineered features ────────────────────────────────────────────
        if engineered:
            with st.expander(f"⚙️ Engineered Features ({len(engineered)} features)", expanded=False):
                st.markdown("Composite and PCA-derived features from Notebook 3.")
                eng_cols = st.columns(3)
                for idx, feat in enumerate(engineered):
                    input_vals[feat] = eng_cols[idx % 3].number_input(
                        feat, value=0.0, step=0.01, format="%.4f",
                    )

        # ── Predict button ─────────────────────────────────────────────────
        st.markdown("")
        col_btn, col_fill = st.columns([1, 3])
        predict_btn = col_btn.button("🔮 Predict AMS Risk", type="primary", use_container_width=True)

        if predict_btn:
            # Build feature vector in correct order
            X_input = np.array([[input_vals.get(f, 0.0) for f in feat_names]], dtype=float)

            # Impute → scale
            X_imp  = imputer.transform(X_input)
            X_sc   = scaler.transform(X_imp)

            # Predict
            prob_ams_pos = model.predict_proba(X_sc)[0, 1]
            pred_label   = int(model.predict(X_sc)[0])

            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

            res_col, gauge_col = st.columns([1, 1])

            with res_col:
                if pred_label == 1:
                    st.markdown(f"""
<div class="risk-high">
  <h2 style="color:#D62728; margin:0">⚠️ AMS Positive</h2>
  <p style="font-size:1.1rem; margin:0.4rem 0 0">Subject predicted <b>susceptible</b> to Acute Mountain Sickness</p>
  <p style="font-size:2rem; font-weight:700; color:#D62728; margin:0.3rem 0 0">{prob_ams_pos*100:.1f}%</p>
  <p style="font-size:0.85rem; color:#666; margin:0">AMS+ probability</p>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="risk-low">
  <h2 style="color:#1F77B4; margin:0">✅ AMS Negative</h2>
  <p style="font-size:1.1rem; margin:0.4rem 0 0">Subject predicted <b>resistant</b> to Acute Mountain Sickness</p>
  <p style="font-size:2rem; font-weight:700; color:#1F77B4; margin:0.3rem 0 0">{prob_ams_pos*100:.1f}%</p>
  <p style="font-size:0.85rem; color:#666; margin:0">AMS+ probability</p>
</div>
""", unsafe_allow_html=True)

                st.markdown("")
                st.markdown(f"""
<div class="metric-card">
  <b>Decision threshold:</b> 0.50 &nbsp;|&nbsp;
  <b>P(AMS+):</b> {prob_ams_pos:.4f} &nbsp;|&nbsp;
  <b>P(AMS-):</b> {1-prob_ams_pos:.4f}
</div>
""", unsafe_allow_html=True)

            with gauge_col:
                # Probability gauge chart
                fig, ax = plt.subplots(figsize=(4, 3.5))
                bar_color = "#D62728" if pred_label == 1 else "#1F77B4"
                ax.barh(["AMS−", "AMS+"],
                        [1 - prob_ams_pos, prob_ams_pos],
                        color=["#1F77B4", "#D62728"], alpha=0.8, edgecolor="white", height=0.5)
                ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.5, label="Threshold (0.5)")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", fontsize=10)
                ax.set_title("Prediction Probability", fontsize=11, fontweight="bold")
                ax.legend(fontsize=8)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── Feature contributions (LR coefficients × scaled values) ───
            if hasattr(model, "coef_"):
                st.markdown('<div class="section-header">🔍 Feature Contributions (LR Coefficients)</div>', unsafe_allow_html=True)
                coefs   = model.coef_[0]
                contrib = coefs * X_sc[0]
                contrib_df = pd.DataFrame({
                    "Feature":      feat_names,
                    "Scaled Value": X_sc[0],
                    "Coefficient":  coefs,
                    "Contribution": contrib,
                }).sort_values("Contribution", key=abs, ascending=False).head(15)

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                colors = ["#D62728" if v > 0 else "#1F77B4" for v in contrib_df["Contribution"].values[::-1]]
                ax2.barh(range(len(contrib_df)), contrib_df["Contribution"].values[::-1],
                         color=colors, alpha=0.85, edgecolor="white")
                ax2.set_yticks(range(len(contrib_df)))
                ax2.set_yticklabels(
                    [f[:28] for f in contrib_df["Feature"].values[::-1]], fontsize=8
                )
                ax2.axvline(0, color="grey", lw=0.8)
                ax2.set_xlabel("Contribution to log-odds (coef × scaled value)", fontsize=9)
                ax2.set_title("Top 15 Feature Contributions for This Subject\n(red = increases AMS+ risk, blue = decreases risk)",
                              fontsize=10, fontweight="bold")
                for spine in ["top", "right"]:
                    ax2.spines[spine].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

                st.dataframe(
                    contrib_df[["Feature", "Scaled Value", "Coefficient", "Contribution"]].round(4),
                    use_container_width=True,
                )

    else:
        st.info("📂 Please upload `final_model_LR.pkl` above to enable predictions.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Batch prediction via CSV / Excel
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Batch Prediction — CSV or Excel Upload</div>', unsafe_allow_html=True)
    st.markdown("""
Upload a **CSV or Excel file** where:
- **Column 1** → `subject_id` (non-negative integer, unique per row)
- **Remaining columns** → feature values (one column per model feature)
- **Each row** → one subject

Subjects with **any missing cell** will be **flagged and skipped** — predictions are only made for complete records.
""")

    bundle2 = get_model_bundle()

    if bundle2 is None:
        st.markdown("""
<div class="info-box">
No model found in <code>models/final_model_LR.pkl</code>. Upload it below.
</div>
""", unsafe_allow_html=True)
        uploaded_model2 = st.file_uploader("Upload `final_model_LR.pkl`", type=["pkl"], key="model2")
        if uploaded_model2:
            bundle2 = pickle.load(uploaded_model2)

    if bundle2:
        model2      = bundle2["model"]
        imputer2    = bundle2["imputer"]
        scaler2     = bundle2["scaler"]
        feat_names2 = bundle2["feat_names"]

        # ── Step 1: Template download ──────────────────────────────────────
        st.markdown("### Step 1 — Download the Input Template")
        st.markdown("Fill in this template and upload it in Step 2. Column order must be: `subject_id` then all feature columns.")

        template_df = pd.DataFrame(columns=["subject_id"] + feat_names2)
        # Add 3 example empty rows
        example_rows = pd.DataFrame(
            [[i] + [None] * len(feat_names2) for i in range(1, 4)],
            columns=["subject_id"] + feat_names2,
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

        # Excel template
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

        st.markdown(f"Template has **1 + {len(feat_names2)} columns** (subject_id + {len(feat_names2)} features).")

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
                # ── Load file ──────────────────────────────────────────────
                fname = batch_file.name.lower()
                if fname.endswith(".csv"):
                    raw_df = pd.read_csv(batch_file)
                else:
                    raw_df = pd.read_excel(batch_file, engine="openpyxl")

                st.markdown(f"**Loaded:** {len(raw_df)} rows × {len(raw_df.columns)} columns")

                # ── Validate subject_id column ─────────────────────────────
                if "subject_id" not in raw_df.columns:
                    st.error("❌ Column `subject_id` not found. First column must be named `subject_id`.")
                    st.stop()

                # Check subject_id is non-negative integer
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

                # Check for duplicate subject IDs
                dupes = raw_df["subject_id"][raw_df["subject_id"].duplicated()].tolist()
                if dupes:
                    st.warning(f"⚠️ Duplicate subject_id values found: {dupes}. Each row should be a unique subject.")

                # ── Check which feature columns are present ────────────────
                cols_in_file    = [c for c in raw_df.columns if c != "subject_id"]
                missing_feature_cols = [f for f in feat_names2 if f not in raw_df.columns]
                extra_cols      = [c for c in cols_in_file if c not in feat_names2]

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

                # ── Per-subject missing value audit ────────────────────────
                feat_df = raw_df[feat_names2].copy()

                # Coerce to numeric — non-numeric cells become NaN
                for col in feat_names2:
                    feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

                missing_per_subject = feat_df.isnull().sum(axis=1)  # Series: index = row
                missing_per_col     = feat_df.isnull().sum(axis=0)  # Series: missing count per feature

                complete_mask   = missing_per_subject == 0
                incomplete_mask = ~complete_mask

                n_total      = len(raw_df)
                n_complete    = complete_mask.sum()
                n_incomplete  = incomplete_mask.sum()

                # ── Summary banner ─────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Data Quality Report")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Subjects", n_total)
                kpi2.metric("✅ Complete (will predict)", n_complete)
                kpi3.metric("⚠️ Incomplete (flagged)", n_incomplete,
                            delta=f"-{n_incomplete}" if n_incomplete > 0 else None,
                            delta_color="inverse")
                kpi4.metric("Features Required", len(feat_names2))

                # ── Flagged subjects detail ────────────────────────────────
                if n_incomplete > 0:
                    st.markdown("#### ⚠️ Flagged Subjects — Missing Data")
                    flagged_df = raw_df[incomplete_mask][["subject_id"]].copy()
                    flagged_df["Missing_Count"] = missing_per_subject[incomplete_mask].values
                    flagged_df["Missing_Features"] = [
                        ", ".join(feat_names2[j] for j in range(len(feat_names2))
                                  if pd.isnull(feat_df.iloc[i, j]))
                        for i in raw_df.index[incomplete_mask]
                    ]
                    flagged_df = flagged_df.reset_index(drop=True)

                    st.dataframe(
                        flagged_df.style.applymap(
                            lambda v: "background-color: #fff3cd; color: #856404;",
                            subset=["Missing_Count"],
                        ),
                        use_container_width=True,
                    )

                    # Show which features have the most missing values
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

                # ── Run predictions on complete subjects only ──────────────
                if n_complete == 0:
                    st.error("❌ No subjects with complete data. Please fix missing values and re-upload.")
                else:
                    st.markdown("---")
                    st.markdown(f"### 🔮 Predictions — {n_complete} Complete Subject(s)")

                    complete_df = raw_df[complete_mask].copy()
                    X_complete  = feat_df[complete_mask].values.astype(float)

                    # Scale only (no imputation needed — data is complete)
                    X_sc = scaler2.transform(X_complete)

                    probs = model2.predict_proba(X_sc)[:, 1]
                    preds = model2.predict(X_sc)

                    def risk_level(p):
                        if p < 0.30:   return "🟢 Low"
                        elif p < 0.50: return "🟡 Moderate"
                        elif p < 0.70: return "🟠 High"
                        else:          return "🔴 Very High"

                    results_df = pd.DataFrame({
                        "subject_id":      complete_df["subject_id"].values,
                        "AMS_Prediction":  ["AMS+" if p == 1 else "AMS-" for p in preds],
                        "P(AMS+)":         probs.round(4),
                        "P(AMS-)":         (1 - probs).round(4),
                        "Risk_Level":      [risk_level(p) for p in probs],
                        "Status":          ["✅ Predicted"] * n_complete,
                    })

                    # Style the results table
                    def style_prediction(val):
                        if val == "AMS+":
                            return "color: #D62728; font-weight: bold"
                        elif val == "AMS-":
                            return "color: #1F77B4; font-weight: bold"
                        return ""

                    st.dataframe(
                        results_df.style.map(style_prediction, subset=["AMS_Prediction"]),
                        use_container_width=True,
                        height=min(400, 60 + 35 * n_complete),
                    )

                    # ── Summary metrics ────────────────────────────────────
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

                    # ── Visualisations ─────────────────────────────────────
                    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))

                    # Plot 1: Prediction count
                    ax = axes3[0]
                    labels_plot = []
                    counts_plot = []
                    colors_plot = []
                    if n_pos > 0:
                        labels_plot.append("AMS+"); counts_plot.append(n_pos); colors_plot.append("#D62728")
                    if n_neg > 0:
                        labels_plot.append("AMS-"); counts_plot.append(n_neg); colors_plot.append("#1F77B4")
                    bars = ax.bar(labels_plot, counts_plot, color=colors_plot, alpha=0.85,
                                  edgecolor="white", width=0.5)
                    for bar, cnt in zip(bars, counts_plot):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                                str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=12)
                    ax.set_title("Prediction Distribution", fontweight="bold", fontsize=11)
                    ax.set_ylabel("Number of Subjects")
                    ax.set_ylim(0, max(counts_plot) * 1.25)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    # Plot 2: Probability bar per subject
                    ax = axes3[1]
                    subj_labels = [str(s) for s in results_df["subject_id"].values]
                    bar_colors  = ["#D62728" if p == 1 else "#1F77B4" for p in preds]
                    ax.bar(subj_labels, probs, color=bar_colors, alpha=0.82, edgecolor="white")
                    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="Threshold (0.5)")
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Subject ID")
                    ax.set_ylabel("P(AMS+)")
                    ax.set_title("AMS+ Probability per Subject", fontweight="bold", fontsize=11)
                    ax.legend(fontsize=8)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    # Plot 3: Probability histogram
                    ax = axes3[2]
                    if n_complete > 1:
                        ax.hist(probs, bins=min(8, n_complete), color="#4C72B0",
                                alpha=0.75, edgecolor="white")
                    else:
                        ax.bar([probs[0]], [1], color="#4C72B0", alpha=0.75, width=0.05)
                    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Threshold")
                    ax.set_xlabel("P(AMS+)")
                    ax.set_ylabel("Count")
                    ax.set_title("Probability Distribution", fontweight="bold", fontsize=11)
                    ax.legend(fontsize=8)
                    for s in ["top", "right"]: ax.spines[s].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close()

                    # ── Full output table (predicted + flagged together) ───
                    st.markdown("### 📋 Full Subject Report (All Subjects)")

                    # Build combined table
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
                        lambda s: "✅ Predicted" if s in pred_map else "⚠️ Skipped (incomplete data)"
                    )

                    def style_status(val):
                        if "Skipped" in str(val):
                            return "background-color: #fff3cd; color: #856404; font-weight: bold"
                        elif "Predicted" in str(val):
                            return "background-color: #d4edda; color: #155724; font-weight: bold"
                        return ""

                    st.dataframe(
                        all_report.style
                            .applymap(style_prediction, subset=["AMS_Prediction"])
                            .applymap(style_status, subset=["Status"]),
                        use_container_width=True,
                    )

                    # ── Downloads ──────────────────────────────────────────
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
# TAB 3 — Model information
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Model Architecture & Validation</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Pipeline Summary")
        st.markdown("""
| Stage | Detail |
|---|---|
| **Model** | Logistic Regression (`class_weight='balanced'`) |
| **Solver** | SAGA |
| **Regularisation** | L1 / L2 (selected via inner CV) |
| **Imputation** | Median (RobustScaler) |
| **Scaling** | RobustScaler |
| **Evaluation** | Nested LOOCV (outer: LOO, inner: StratifiedKFold-3) |
| **Feature matrix** | BL_full — n=21 subjects |
| **Feature count** | 67 (60 genes + 6 physiology + sex) |
| **Class imbalance** | AMS+: 16 / AMS-: 5 (balanced weight) |
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

    # Visualise feature importance from model coefficients if loaded
    bundle3 = get_model_bundle()
    if bundle3 and hasattr(bundle3["model"], "coef_"):
        st.markdown('<div class="section-header">📊 Model Coefficients (Global Feature Importance)</div>', unsafe_allow_html=True)
        model3     = bundle3["model"]
        feat_names3 = bundle3["feat_names"]
        coefs      = model3.coef_[0]
        coef_df    = pd.DataFrame({
            "Feature": feat_names3,
            "Coefficient": coefs,
            "|Coefficient|": np.abs(coefs),
        }).sort_values("|Coefficient|", ascending=False).head(20)

        fig4, ax4 = plt.subplots(figsize=(9, 6))
        colors4 = ["#D62728" if v > 0 else "#1F77B4" for v in coef_df["Coefficient"].values[::-1]]
        ax4.barh(range(len(coef_df)), coef_df["Coefficient"].values[::-1],
                 color=colors4, alpha=0.82, edgecolor="white")
        ax4.set_yticks(range(len(coef_df)))
        ax4.set_yticklabels([f[:30] for f in coef_df["Feature"].values[::-1]], fontsize=8)
        ax4.axvline(0, color="grey", lw=0.8)
        ax4.set_xlabel("Coefficient (positive = increases AMS+ risk)", fontsize=9)
        ax4.set_title("Top 20 LR Feature Coefficients\n(red = AMS+ risk ↑ | blue = AMS+ risk ↓)",
                      fontsize=10, fontweight="bold")
        patches = [
            mpatches.Patch(color="#D62728", label="Increases AMS+ risk"),
            mpatches.Patch(color="#1F77B4", label="Decreases AMS+ risk"),
        ]
        ax4.legend(handles=patches, fontsize=8)
        for spine in ["top", "right"]:
            ax4.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.dataframe(coef_df.reset_index(drop=True).round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📋 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
This app deploys the **Logistic Regression** model developed for **Acute Mountain Sickness (AMS) susceptibility prediction** using pre-exposure baseline transcriptomic and physiological data.

### Project Pipeline

| Notebook | Purpose |
|---|---|
| **NB1** — Data Integration & Cleaning | Merge physiology + gene expression, compute delta features |
| **NB2** — Baseline Biomarker & Delta Correlation | Differential expression, gene–physiology Spearman correlations, bootstrap stability |
| **NB3** — Feature Engineering | Multi-criteria gene selection, composite score construction, PCA, scaling |
| **NB4** — Classification | Nested LOOCV, RF + LR training, SHAP explainability, model serialisation |
| **NB5** — Visualisation & Interpretation | Centralised plots and biological interpretation |

### Cohort
- **n=21** subjects for baseline (BL) matrix — includes subjects 3 & 4 (intact baseline)
- **n=19** for delta/combined matrices — subjects 3 & 4 excluded (corrupted D1 data)
- **AMS+:** 16 | **AMS-:** 5 (BL matrix)

### Applications
- High-altitude military deployment screening
- Expedition medicine pre-screening
- Precision healthcare for altitude-related conditions

### Limitations
- Small sample size (n=21) — validate in an independent cohort before deployment
- Class imbalance (16:5) — balanced weights applied; AUC and MCC preferred over accuracy
- Gene expression requires RNA-seq profiling — operational feasibility depends on turnaround time
""")

    st.markdown("""
<div class="warning-box">
⚠️ <b>Disclaimer:</b> This tool is intended for research and educational purposes only. 
It should not be used as the sole basis for clinical or operational medical decisions.
Validate with an independent cohort and consult qualified medical professionals before any deployment.
</div>
""", unsafe_allow_html=True)
