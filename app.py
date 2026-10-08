"""app.py - LendIQ: Smart Loan Eligibility & Risk Intelligence Engine.
Modern, light-themed, user-friendly UI for loan risk assessment.
"""
import json
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from features import add_features, FEATURES
from risk import risk_level

# 1. Page Configuration
st.set_page_config(
    page_title="LendIQ | Smart Loan Eligibility & Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Light-Themed Custom Styling
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Overall background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Top Brand Hero */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        right: -30px;
        bottom: -30px;
        width: 180px;
        height: 180px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #E0E7FF;
        margin: 0;
        font-weight: 500;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Clean Card Container */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }
    .card-header {
        font-size: 16px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Decision Result Banners */
    .decision-approved {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 24px;
        color: #064E3B;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.2);
        margin: 20px 0;
    }
    .decision-rejected {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 2px solid #F43F5E;
        border-radius: 16px;
        padding: 24px;
        color: #881337;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(244, 63, 94, 0.2);
        margin: 20px 0;
    }
    .decision-title {
        font-size: 24px;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -0.3px;
    }
    .decision-subtitle {
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 14px;
    }
    .risk-low {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
    }
    .risk-medium {
        background-color: #FEF08A;
        color: #713F12;
        border: 1px solid #EAB308;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
    }

    /* KPI metric tiles */
    .kpi-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Warning Flags Pill */
    .warning-pill {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        color: #92400E;
        font-size: 13px;
        font-weight: 600;
    }

    /* Input styling enhancements */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        border-radius: 8px !important;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        padding: 24px 0 10px;
        color: #94A3B8;
        font-size: 13px;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 3. Model & Data Loader with Caching
@st.cache_resource
def load_all_artifacts():
    final = joblib.load("models/final_model.joblib")
    tuned_base = joblib.load("models/tuned_models.joblib")
    
    # Also load extra boosting models if available
    extra_models = {}
    if os.path.exists("models/tuned_extra_models.joblib"):
        try:
            extra_models = joblib.load("models/tuned_extra_models.joblib")
        except Exception:
            pass

    # Merge all models
    all_models = {**tuned_base, **extra_models}

    # Load settings
    with open("models/final_settings.json") as f:
        settings = json.load(f)

    # Load metrics (extended if available)
    if os.path.exists("models/metrics_extended.csv"):
        metrics_df = pd.read_csv("models/metrics_extended.csv")
    else:
        metrics_df = pd.read_csv("models/metrics.csv")

    feature_imp_df = None
    if os.path.exists("models/feature_importance.csv"):
        feature_imp_df = pd.read_csv("models/feature_importance.csv")

    return final, all_models, settings, metrics_df, feature_imp_df


final_model, all_models, settings, metrics_df, feature_imp_df = load_all_artifacts()
THRESHOLD = float(settings.get("threshold", 0.49))
BEST_MODEL_NAME = settings.get("best_model", "AdaBoost")

# 4. Header Banner
st.markdown(
    f"""
    <div class="hero-banner">
        <span class="hero-badge">Production ML Underwriting</span>
        <h1 class="hero-title">LendIQ | Smart Loan Eligibility & Risk Intelligence</h1>
        <p class="hero-subtitle">
            Automated credit decisioning powered by calibrated machine learning ensembles ({len(all_models)} algorithms). 
            Benchmarked against state-of-the-art boosting architectures.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 5. Application Tabs
tab_predict, tab_compare, tab_eda, tab_pipeline = st.tabs([
    "🎯 Loan Eligibility Checker",
    "📊 Model Benchmark (10 Algorithms)",
    "📈 Exploratory Data Analysis",
    "⚙️ Pipeline & Underwriting Logic"
])

# ==============================================================================
# TAB 1: PREDICTION & RISK ASSESSMENT
# ==============================================================================
with tab_predict:
    st.markdown("### Underwrite a New Loan Application")
    st.markdown("Fill out the applicant's profile below, or choose a pre-filled persona to test edge cases.")

    # Preset selection to delight users & evaluators
    preset_cols = st.columns([1.2, 1, 1, 1])
    preset_cols[0].markdown("**💡 Quick Test Presets:**")
    
    if "form_gender" not in st.session_state:
        st.session_state.form_gender = "Male"
        st.session_state.form_married = "Yes"
        st.session_state.form_dependents = "0"
        st.session_state.form_education = "Graduate"
        st.session_state.form_self_employed = "No"
        st.session_state.form_applicant_income = 5800
        st.session_state.form_coapplicant_income = 2100
        st.session_state.form_loan_amount = 140
        st.session_state.form_term = 360
        st.session_state.form_credit_history = 1
        st.session_state.form_property_area = "Semiurban"

    if preset_cols[1].button("🟢 Prime Salaried", use_container_width=True):
        st.session_state.form_gender = "Male"
        st.session_state.form_married = "Yes"
        st.session_state.form_dependents = "1"
        st.session_state.form_education = "Graduate"
        st.session_state.form_self_employed = "No"
        st.session_state.form_applicant_income = 7500
        st.session_state.form_coapplicant_income = 3000
        st.session_state.form_loan_amount = 150
        st.session_state.form_term = 360
        st.session_state.form_credit_history = 1
        st.session_state.form_property_area = "Semiurban"
        st.rerun()

    if preset_cols[2].button("🟡 High DTI Borderline", use_container_width=True):
        st.session_state.form_gender = "Female"
        st.session_state.form_married = "No"
        st.session_state.form_dependents = "2"
        st.session_state.form_education = "Graduate"
        st.session_state.form_self_employed = "Yes"
        st.session_state.form_applicant_income = 3200
        st.session_state.form_coapplicant_income = 0
        st.session_state.form_loan_amount = 220
        st.session_state.form_term = 180
        st.session_state.form_credit_history = 1
        st.session_state.form_property_area = "Rural"
        st.rerun()

    if preset_cols[3].button("🔴 Poor Credit History", use_container_width=True):
        st.session_state.form_gender = "Male"
        st.session_state.form_married = "Yes"
        st.session_state.form_dependents = "3+"
        st.session_state.form_education = "Not Graduate"
        st.session_state.form_self_employed = "No"
        st.session_state.form_applicant_income = 4000
        st.session_state.form_coapplicant_income = 0
        st.session_state.form_loan_amount = 180
        st.session_state.form_term = 360
        st.session_state.form_credit_history = 0
        st.session_state.form_property_area = "Urban"
        st.rerun()

    # Form with clean 3-column grouping
    with st.form("applicant_evaluation_form"):
        col_demo, col_fin, col_loan = st.columns(3)

        with col_demo:
            st.markdown("##### 👤 Applicant Demographics")
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                index=["Male", "Female"].index(st.session_state.form_gender)
            )
            married = st.selectbox(
                "Marital Status", ["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.form_married)
            )
            dependents = st.selectbox(
                "Number of Dependents", ["0", "1", "2", "3+"],
                index=["0", "1", "2", "3+"].index(st.session_state.form_dependents)
            )
            education = st.selectbox(
                "Education Level", ["Graduate", "Not Graduate"],
                index=["Graduate", "Not Graduate"].index(st.session_state.form_education)
            )
            self_employed = st.selectbox(
                "Self Employed", ["No", "Yes"],
                index=["No", "Yes"].index(st.session_state.form_self_employed)
            )

        with col_fin:
            st.markdown("##### 💵 Financial Baseline")
            applicant_income = st.number_input(
                "Applicant Monthly Income ($)",
                min_value=0, max_value=1_000_000,
                value=int(st.session_state.form_applicant_income),
                step=250
            )
            coapplicant_income = st.number_input(
                "Co-Applicant Monthly Income ($)",
                min_value=0, max_value=1_000_000,
                value=int(st.session_state.form_coapplicant_income),
                step=250
            )
            credit_history = st.selectbox(
                "Credit Bureau History",
                [1, 0],
                format_func=lambda x: "✅ Good / Meets Guidelines (1)" if x == 1 else "⚠️ Bad / Delinquent Records (0)",
                index=[1, 0].index(st.session_state.form_credit_history)
            )
            
            # Live financial calculation preview
            total_income_preview = applicant_income + coapplicant_income
            st.caption(f"**Total Household Income:** ${total_income_preview:,.0f} / month")

        with col_loan:
            st.markdown("##### 📋 Loan Request Specs")
            loan_amount = st.number_input(
                "Loan Amount (in Thousands, $K)",
                min_value=5, max_value=2000,
                value=int(st.session_state.form_loan_amount),
                step=5
            )
            loan_term = st.selectbox(
                "Loan Term (Months)",
                [360, 300, 240, 180, 120, 84, 60, 36, 12],
                index=[360, 300, 240, 180, 120, 84, 60, 36, 12].index(st.session_state.form_term)
            )
            property_area = st.selectbox(
                "Property Geography",
                ["Urban", "Semiurban", "Rural"],
                index=["Urban", "Semiurban", "Rural"].index(st.session_state.form_property_area)
            )
            
            # Estimated EMI preview
            est_monthly_emi = (loan_amount * 1000) / max(loan_term, 1)
            est_dti = (est_monthly_emi / (total_income_preview + 1)) * 100
            st.caption(f"**Estimated Monthly EMI:** ${est_monthly_emi:,.0f} / month")
            st.caption(f"**Estimated Debt-to-Income:** {est_dti:.1f}%")

        submitted = st.form_submit_button("🔍 Run Multi-Model Risk Evaluation", type="primary", use_container_width=True)

    if submitted:
        # Build evaluation dataframe
        raw_input = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": str(dependents),
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
        }])

        # Apply feature engineering
        engineered_df = add_features(raw_input)
        X_eval = engineered_df[FEATURES]

        # Calculate final calibrated prediction
        calibrated_prob = float(final_model.predict_proba(X_eval)[0, 1])
        dti_ratio = float(X_eval["Debt_to_Income"].iloc[0])
        emi_val = float(X_eval["EMI"].iloc[0])
        total_inc = float(X_eval["Total_Income"].iloc[0])
        
        # Risk classification & warning rules
        risk_label, warning_flags = risk_level(calibrated_prob, credit_history, dti_ratio)
        is_approved = calibrated_prob >= THRESHOLD

        # Hero Decision Banner
        if is_approved:
            st.markdown(
                f"""
                <div class="decision-approved">
                    <div style="font-size: 32px;">✅</div>
                    <div class="decision-title">APPLICATION RECOMMENDED FOR APPROVAL</div>
                    <div class="decision-subtitle">
                        Calibrated Approval Probability: <strong>{calibrated_prob:.1%}</strong> 
                        (Optimal Decision Cutoff: {THRESHOLD:.0%})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="decision-rejected">
                    <div style="font-size: 32px;">🛑</div>
                    <div class="decision-title">APPLICATION DECLINED / HIGH CREDIT RISK</div>
                    <div class="decision-subtitle">
                        Calibrated Approval Probability: <strong>{calibrated_prob:.1%}</strong> 
                        (Below Required Cutoff: {THRESHOLD:.0%})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Key Metrics Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Final Decision</div>
                    <div class="kpi-value" style="color: {'#059669' if is_approved else '#E11D48'};">
                        {'Eligible' if is_approved else 'Not Eligible'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Approval Confidence</div>
                    <div class="kpi-value" style="color: #2563EB;">{calibrated_prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi3:
            badge_class = "risk-low" if "Low" in risk_label else ("risk-medium" if "Medium" in risk_label else "risk-high")
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Risk Category</div>
                    <div style="margin-top: 6px;">
                        <span class="risk-badge {badge_class}">{risk_label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Debt-To-Income (DTI)</div>
                    <div class="kpi-value" style="color: {'#E11D48' if dti_ratio > 0.4 else '#0F172A'};">
                        {dti_ratio:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Progress / Confidence Bar with Threshold marker
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**Approval Probability Meter:** {calibrated_prob:.1%} vs Threshold ({THRESHOLD:.0%})")
        st.progress(calibrated_prob)

        # Warnings / Underwriter flags
        if warning_flags:
            st.markdown("##### ⚠️ Underwriter Warning Flags")
            for flag in warning_flags:
                st.markdown(f'<div class="warning-pill">⚠️ {flag}</div>', unsafe_allow_html=True)

        # Multi-Model Consensus Breakdown
        st.markdown("---")
        st.markdown("#### 🤖 Multi-Model Consensus (All 10 Algorithms)")
        st.markdown("Compare predictions across classical and state-of-the-art boosting algorithms for this applicant:")

        model_results = []
        for name, model in all_models.items():
            try:
                p = float(model.predict_proba(X_eval)[0, 1])
                verdict = "Approve" if p >= 0.5 else "Reject"
                model_results.append({
                    "Model": name,
                    "Approval Probability": p,
                    "Recommendation": verdict,
                    "Is Best Model": "⭐ Best" if name == BEST_MODEL_NAME else ""
                })
            except Exception as e:
                pass

        consensus_df = pd.DataFrame(model_results).sort_values("Approval Probability", ascending=False)
        approval_count = (consensus_df["Approval Probability"] >= 0.5).sum()
        total_count = len(consensus_df)

        st.info(
            f"**Consensus Verdict:** **{approval_count} out of {total_count} models** recommend **Approval** for this applicant profile."
        )

        col_chart, col_tbl = st.columns([1.5, 1])
        with col_chart:
            # Bar chart of model probabilities
            chart_df = consensus_df.set_index("Model")[["Approval Probability"]]
            st.bar_chart(chart_df, color="#2563EB")
        
        with col_tbl:
            styled_consensus = consensus_df.copy()
            styled_consensus["Approval Probability"] = styled_consensus["Approval Probability"].map(lambda x: f"{x:.1%}")
            st.dataframe(styled_consensus, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 2: MODEL BENCHMARK (10 ALGORITHMS)
# ==============================================================================
with tab_compare:
    st.markdown("### 📊 Comprehensive Model Benchmark & Evaluation")
    st.markdown(
        "Performance on the hold-out test set (123 unseen applicants, 20% stratified test split). "
        "All models evaluated under strictly identical cross-validation folds and feature pipelines."
    )

    # Top highlight cards
    b1, b2, b3, b4 = st.columns(4)
    best_roc_row = metrics_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    best_acc_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_f1_row = metrics_df.sort_values("F1", ascending=False).iloc[0]

    b1.metric("Top ROC-AUC Champion", f"{best_roc_row['Model']}", f"{best_roc_row['ROC-AUC']:.3f} AUC")
    b2.metric("Highest Test Accuracy", f"{best_acc_row['Model']}", f"{best_acc_row['Accuracy']:.1%}")
    b3.metric("Highest Test F1 Score", f"{best_f1_row['Model']}", f"{best_f1_row['F1']:.3f}")
    b4.metric("Total Algorithms Evaluated", f"{len(metrics_df)} Models", "State-of-the-Art")

    st.markdown("#### Detailed Performance Metrics Table")
    # Clean up display of metrics
    display_metrics = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC"]:
        if col in display_metrics.columns:
            display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "—")
    
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)

    # Diagnostic Visualizations
    st.markdown("---")
    st.markdown("#### Diagnostic Evaluation Plots")
    plot_col1, plot_col2 = st.columns([1.2, 1])

    with plot_col1:
        st.markdown("##### 📈 Receiver Operating Characteristic (ROC Curves)")
        if os.path.exists("reports/04_roc_curves.png"):
            st.image("reports/04_roc_curves.png", use_container_width=True)
        else:
            st.info("ROC Curve image not found in reports/")

    with plot_col2:
        st.markdown(f"##### 🎯 Confusion Matrix ({BEST_MODEL_NAME})")
        if os.path.exists("reports/04_confusion_matrix.png"):
            st.image("reports/04_confusion_matrix.png", use_container_width=True)
        else:
            st.info("Confusion matrix image not found in reports/")

    # Feature Importance Section
    if feature_imp_df is not None:
        st.markdown("---")
        st.markdown("#### 🔬 Permutation Feature Importance (Unseen Test Set)")
        st.markdown(
            "Measures how much the model's test ROC-AUC drops when values of each feature are randomly shuffled. "
            "Higher importance indicates features critical for underwriting decisions."
        )
        f_col1, f_col2 = st.columns([1.4, 1])
        with f_col1:
            top_features = feature_imp_df.head(10).set_index("Feature")
            st.bar_chart(top_features, color="#3B82F6")
        with f_col2:
            st.markdown(
                """
                **Key Underwriting Takeaways:**
                - **Credit_History** is overwhelmingly the single most decisive factor in loan eligibility.
                - **Financial ratios** (`Debt_to_Income`, `EMI`, `Income_to_Loan`) engineered from income and loan terms contribute substantial signal beyond raw income alone.
                - **Property_Area** (particularly Semiurban location) acts as a strong positive indicator for credit recovery.
                """
            )

# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("### 📈 Exploratory Data Analysis & Statistical Distribution")
    st.markdown(
        "Insights derived from the raw historical loan portfolio (614 applicants). "
        "Overall target approval baseline: **68.7% Approved (422) vs 31.3% Rejected (192)**."
    )

    eda1, eda2 = st.columns(2)
    with eda1:
        st.markdown("##### 🔍 Cohort Distributions & Bivariate Count Plots")
        if os.path.exists("reports/01_overview.png"):
            st.image("reports/01_overview.png", use_container_width=True)
        else:
            st.info("Overview plot not found in reports/")

    with eda2:
        st.markdown("##### 🔥 Feature Correlation Heatmap")
        if os.path.exists("reports/01_correlation.png"):
            st.image("reports/01_correlation.png", use_container_width=True)
        else:
            st.info("Correlation plot not found in reports/")

    st.markdown(
        """
        <div class="clean-card">
            <div class="card-header">💡 Key Statistical Findings</div>
            <ul>
                <li><strong>Credit History Asymmetry:</strong> Over 79% of applicants with a clean credit history (Credit_History = 1.0) received loan approval, compared to under 8% with past defaults.</li>
                <li><strong>Income Skewness:</strong> Applicant income exhibits strong positive skewness with extreme outliers up to $81,000/mo. Addressed via logarithmic feature transformations (<code>Log_Total_Income</code>, <code>Log_LoanAmount</code>).</li>
                <li><strong>Regional Disparities:</strong> Semiurban applicants show an approval rate of ~76%, markedly higher than Rural (~61%) and Urban (~65%) applicants.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# TAB 4: METHODOLOGY & PIPELINE
# ==============================================================================
with tab_pipeline:
    st.markdown("### ⚙️ Production Machine Learning Pipeline")
    
    st.markdown(
        """
        <div class="clean-card">
            <div class="card-header">📐 System Architecture & Workflow</div>
            <ol>
                <li><strong>Data Preprocessing & Imputation:</strong>
                    <ul>
                        <li>Numerical columns imputed via <code>SimpleImputer(strategy='median')</code> to resist income outliers.</li>
                        <li>Categorical columns imputed with <code>strategy='most_frequent'</code>.</li>
                        <li>Standardized with <code>StandardScaler</code> and encoded with <code>OneHotEncoder(handle_unknown='ignore')</code>.</li>
                    </ul>
                </li>
                <li><strong>Domain Feature Engineering:</strong>
                    <ul>
                        <li><code>Total_Income = ApplicantIncome + CoapplicantIncome</code></li>
                        <li><code>EMI = LoanAmount / Loan_Amount_Term</code></li>
                        <li><code>Debt_to_Income = (EMI * 1000) / (Total_Income + 1)</code></li>
                        <li><code>Income_to_Loan = Total_Income / (LoanAmount * 1000 + 1)</code></li>
                        <li><code>Log_Total_Income = log1p(Total_Income)</code> and <code>Log_LoanAmount = log1p(LoanAmount)</code></li>
                    </ul>
                </li>
                <li><strong>5-Fold Stratified Cross-Validation:</strong>
                    Hyperparameters tuned with <code>GridSearchCV</code> optimizing for <code>ROC-AUC</code> to preserve performance under class imbalance (68.7% vs 31.3%).
                </li>
                <li><strong>Probability Calibration (Platt / Sigmoid):</strong>
                    Ensemble tree classifiers output distorted, uncalibrated probabilities. Calibrated via <code>CalibratedClassifierCV(method='sigmoid')</code> to yield reliable Brier scores for real-world risk underwriting.
                </li>
                <li><strong>Optimal Decision Thresholding:</strong>
                    Threshold tuned on out-of-fold training predictions to maximize the <code>F1-score</code> (Threshold = 0.49).
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <div class="app-footer">
        LendIQ Loan Risk System &bull; Production Machine Learning &bull; Built with Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)