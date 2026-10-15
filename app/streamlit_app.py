"""
NovaCart AI Business Intelligence System - Executive Streamlit Dashboard
Dual-Engine Machine Learning Application for Pre-Fulfillment Return Risk Prediction,
Expected Profit Forecasting, Explainable AI (SHAP), and Strategic Decision Simulation.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide responsive page layout
st.set_page_config(
    page_title="NovaCart AI Business Intelligence System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Path Resolution
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = ROOT_DIR / "data"

# Import custom feature engineer class to ensure pipeline unpickling compatibility
try:
    from src.feature_engineering import NovaCartFeatureEngineer
except ImportError:
    import sys
    sys.path.append(str(ROOT_DIR))
    from src.feature_engineering import NovaCartFeatureEngineer


# Custom CSS styling for modern executive aesthetics
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Badges */
    .badge-low {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #854d0e;
        color: #facc15;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    /* Section Headers */
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Loads serialized champion pipelines for classification and regression.
    """
    return_model_path = MODELS_DIR / "return_model.pkl"
    profit_model_path = MODELS_DIR / "profit_model.pkl"

    if not return_model_path.exists() or not profit_model_path.exists():
        st.error("Trained models not found. Please execute training scripts first.")
        return None, None

    return_pipe = joblib.load(return_model_path)
    profit_pipe = joblib.load(profit_model_path)
    return return_pipe, profit_pipe


@st.cache_data(show_spinner=False)
def load_report_data():
    """
    Loads JSON evaluation metrics and business insights.
    """
    class_metrics = {}
    reg_metrics = {}
    return_summary = {}
    profit_summary = {}

    cm_path = REPORTS_DIR / "classification_metrics.json"
    rm_path = REPORTS_DIR / "regression_metrics.json"
    rs_path = REPORTS_DIR / "return_evaluation_summary.json"
    ps_path = REPORTS_DIR / "profit_evaluation_summary.json"

    if cm_path.exists():
        with open(cm_path, "r", encoding="utf-8") as f:
            class_metrics = json.load(f)
    if rm_path.exists():
        with open(rm_path, "r", encoding="utf-8") as f:
            reg_metrics = json.load(f)
    if rs_path.exists():
        with open(rs_path, "r", encoding="utf-8") as f:
            return_summary = json.load(f)
    if ps_path.exists():
        with open(ps_path, "r", encoding="utf-8") as f:
            profit_summary = json.load(f)

    return class_metrics, reg_metrics, return_summary, profit_summary


# Initialize Models and Metadata
return_pipeline, profit_pipeline = load_models()
class_metrics, reg_metrics, return_summary, profit_summary = load_report_data()

# ==============================================================================
# SIDEBAR: Operational Parameter Controls
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("NovaCart AI")
    st.caption("Sales & Profitability Dual-Engine Intelligence")
    st.markdown("---")

    st.subheader("🛠️ Order Parameter Inputs")

    region = st.selectbox(
        "Geographic Region",
        options=["South", "West", "North", "East"],
        index=0,
        help="Operating territory where the order is placed.",
    )

    category = st.selectbox(
        "Product Category",
        options=["Furniture", "Home Appliances", "Electronics", "Fashion"],
        index=0,
        help="Merchandise department.",
    )

    sales = st.number_input(
        "Gross Sales ($)",
        min_value=1000.0,
        max_value=150000.0,
        value=48500.0,
        step=500.0,
        help="Total list ticket price before promotional discounts.",
    )

    discount_pct = st.slider(
        "Discount Percentage (%)",
        min_value=0,
        max_value=35,
        value=15,
        step=1,
        help="Concession given to customer (EDA identified >20% as high risk).",
    )

    delivery_days = st.slider(
        "Delivery Transit Days",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Promised or elapsed fulfillment duration (SLA breach >5 days).",
    )

    customer_rating = st.slider(
        "Customer Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.4,
        step=0.1,
        help="User satisfaction score (Rating <3.5 correlates with peak returns).",
    )

    st.markdown("---")
    st.info(
        "💡 **Dual-Model Inference:**\nInputs are dynamically processed through both "
        "the Return Classification engine (XGBoost) and Profit Regressor (Gradient Boosting)."
    )

# Prepare Single Inference Payload DataFrame
input_df = pd.DataFrame(
    [
        {
            "Region": region,
            "Category": category,
            "Sales": sales,
            "Discount_pct": discount_pct,
            "Delivery_Days": delivery_days,
            "Customer_Rating": customer_rating,
        }
    ]
)

# Run Inference
if return_pipeline and profit_pipeline:
    return_prob = float(return_pipeline.predict_proba(input_df)[0][1])
    is_returned = int(return_pipeline.predict(input_df)[0])
    predicted_profit = float(profit_pipeline.predict(input_df)[0])
else:
    return_prob, is_returned, predicted_profit = 0.25, 0, 4500.0

# Post-inference business indicators
expected_margin = (predicted_profit / sales) * 100.0
confidence_score = max(return_prob, 1.0 - return_prob) * 100.0

# Determine Risk Level Category
if return_prob < 0.30:
    risk_level = "Low"
    risk_badge = '<span class="badge-low">🟢 Low Return Risk</span>'
elif return_prob <= 0.60:
    risk_level = "Medium"
    risk_badge = '<span class="badge-medium">🟡 Medium Return Risk</span>'
else:
    risk_level = "High"
    risk_badge = '<span class="badge-high">🔴 High Return Risk</span>'

# Determine Profit Tier Category
if expected_margin < 5.0:
    profit_category = "Low Profit"
    profit_badge = '<span class="badge-high">⚠️ Low Profit Margin (<5%)</span>'
elif expected_margin <= 15.0:
    profit_category = "Medium Profit"
    profit_badge = '<span class="badge-medium">⚖️ Medium Profit Margin (5-15%)</span>'
else:
    profit_category = "High Profit"
    profit_badge = '<span class="badge-low">💎 High Profit Margin (>15%)</span>'

# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================
st.title("🛒 NovaCart AI Business Intelligence System")
st.markdown(
    "Predictive return risk mitigation, dynamic profit margin forecasting, "
    "and explainable AI analytics for executive supply chain optimization."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dual-Engine Order Assessment",
        "🎯 Section 1: Return Risk Prediction",
        "💰 Section 2: Profit Forecasting",
        "🔍 Section 3: Business Insights & XAI",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE DUAL-ENGINE OVERVIEW
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("⚡ Real-Time Operational Decision Matrix")
    st.caption("Live synthesis across pre-fulfillment return probability and expected margin realization.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Status</div>
                <div class="metric-val" style="color: {'#f87171' if is_returned else '#34d399'};">
                    {'Returned' if is_returned else 'Retained'}
                </div>
                <div class="metric-sub">{risk_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Return Probability</div>
                <div class="metric-val">{return_prob:.1%}</div>
                <div class="metric-sub">Confidence: {confidence_score:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Profit</div>
                <div class="metric-val">${predicted_profit:,.2f}</div>
                <div class="metric-sub">Net Margin: {expected_margin:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Profitability Tier</div>
                <div class="metric-val" style="font-size: 1.3rem;">{profit_category}</div>
                <div class="metric-sub">{profit_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operational Gauge Charts
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_g1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=return_prob * 100,
                title={"text": "Return Likelihood Index (%)", "font": {"size": 15, "weight": "bold"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if return_prob > 0.6 else "#f59e0b" if return_prob >= 0.3 else "#10b981"},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [30, 60], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50},
                },
            )
        )
        fig_g1.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_g1, use_container_width=True)

    with col_g2:
        fig_g2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=expected_margin,
                number={"suffix": "%"},
                title={"text": "Expected Profit Margin (%)", "font": {"size": 15, "weight": "bold"}},
                gauge={
                    "axis": {"range": [0, 25]},
                    "bar": {"color": "#10b981" if expected_margin > 15 else "#f59e0b" if expected_margin >= 5 else "#ef4444"},
                    "steps": [
                        {"range": [0, 5], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [5, 15], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [15, 25], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                },
            )
        )
        fig_g2.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_g2, use_container_width=True)

    # Strategic Action Callout
    st.markdown("### 📋 Automated Fulfillment Policy Directive")
    if return_prob > 0.60:
        st.error(
            f"🚨 **HIGH RISK ALERT:** This order has a **{return_prob:.1%} return probability**. "
            f"Key risk triggers: Discount ({discount_pct}%), Delivery Transit ({delivery_days} days), Rating ({customer_rating}). "
            f"**Recommended Action:** Require customer confirmation before dispatch or offer dedicated priority expedited shipping."
        )
    elif return_prob >= 0.30:
        st.warning(
            f"⚠️ **ELEVATED MONITORING:** Order return probability is **{return_prob:.1%}**. "
            f"Ensure carrier adheres strictly to the {delivery_days}-day fulfillment SLA."
        )
    else:
        st.success(
            f"✅ **PRIME FULFILLMENT:** Low risk profile ({return_prob:.1%} return risk). "
            f"Proceed with standard automated fulfillment pipeline."
        )

# ------------------------------------------------------------------------------
# TAB 2: SECTION 1 - RETURN RISK PREDICTION
# ------------------------------------------------------------------------------
with tab2:
    st.header("🎯 Section 1: Pre-Fulfillment Return Risk Prediction")
    st.markdown("Automated classification engine trained on historical operational orders.")

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown("#### Inference Breakdown")
        summary_table = pd.DataFrame(
            {
                "Parameter": ["Region", "Category", "Gross Sales", "Discount Rate", "Delivery Duration", "Customer Rating"],
                "Input Value": [region, category, f"${sales:,.2f}", f"{discount_pct}%", f"{delivery_days} days", f"{customer_rating} / 5.0"],
            }
        )
        st.dataframe(summary_table, hide_index=True, use_container_width=True)

        st.markdown("#### Classification Output")
        out_table = pd.DataFrame(
            {
                "Output Metric": ["Predicted Class", "Return Probability", "Retention Probability", "Confidence Score", "Risk Tier"],
                "Result": [
                    "Returned" if is_returned else "Retained (Not Returned)",
                    f"{return_prob:.2%}",
                    f"{1 - return_prob:.2%}",
                    f"{confidence_score:.2f}%",
                    risk_level,
                ],
            }
        )
        st.dataframe(out_table, hide_index=True, use_container_width=True)

    with col_r2:
        st.markdown("#### Model Diagnostic Benchmarks")
        if (FIGURES_DIR / "confusion_matrix.png").exists():
            st.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="Champion XGBoost Confusion Matrix (Holdout Test Set)")
        elif (FIGURES_DIR / "roc_curve.png").exists():
            st.image(str(FIGURES_DIR / "roc_curve.png"), caption="XGBoost ROC Curve (AUC = 0.863)")

    st.markdown("---")
    st.subheader("Model Comparison Matrix (Return Classification)")
    if class_metrics:
        rows = []
        for model_name, metrics in class_metrics.items():
            if isinstance(metrics, dict) and "Test Accuracy" in metrics:
                rows.append(
                    {
                        "Algorithm": model_name,
                        "CV ROC-AUC": f"{metrics.get('Best CV ROC-AUC', 0):.4f}",
                        "Test Accuracy": f"{metrics.get('Test Accuracy', 0):.2%}",
                        "Test Precision": f"{metrics.get('Test Precision', 0):.2%}",
                        "Test Recall": f"{metrics.get('Test Recall', 0):.2%}",
                        "Test F1-Score": f"{metrics.get('Test F1 Score', 0):.4f}",
                        "Test ROC-AUC": f"{metrics.get('Test ROC AUC', 0):.4f}",
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: SECTION 2 - PROFIT FORECASTING
# ------------------------------------------------------------------------------
with tab3:
    st.header("💰 Section 2: Expected Profit & Margin Forecasting")
    st.markdown("Regression engine forecasting realized net dollar profit and margin realization.")

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("#### Financial Realization Breakdown")
        discount_dollars = sales * (discount_pct / 100.0)
        net_sales = sales - discount_dollars

        fin_df = pd.DataFrame(
            {
                "Accounting Metric": ["Gross Invoiced Sales", "Discounts Conceded", "Net Realized Sales", "Forecasted Profit", "Expected Margin %"],
                "Amount ($)": [
                    f"${sales:,.2f}",
                    f"-${discount_dollars:,.2f}",
                    f"${net_sales:,.2f}",
                    f"${predicted_profit:,.2f}",
                    f"{expected_margin:.2f}%",
                ],
            }
        )
        st.dataframe(fin_df, hide_index=True, use_container_width=True)

        st.markdown(f"**Profitability Status:** {profit_badge}", unsafe_allow_html=True)
        st.markdown(
            """
            - **High Profit (>15%):** Furniture or premium basket sizes with controlled discounting.
            - **Medium Profit (5-15%):** Home Appliances and standard category baselines.
            - **Low Profit (<5%):** Discounted Fashion and Electronics suffering severe margin compression.
            """
        )

    with col_p2:
        st.markdown("#### Regression Diagnostic Benchmarks")
        if (FIGURES_DIR / "profit_predicted_vs_actual.png").exists():
            st.image(str(FIGURES_DIR / "profit_predicted_vs_actual.png"), caption="Predicted vs. Actual Fit (R² = 0.9899, RMSE = $522.76)")
        elif (FIGURES_DIR / "profit_residuals.png").exists():
            st.image(str(FIGURES_DIR / "profit_residuals.png"), caption="Residual Error Distribution")

    st.markdown("---")
    st.subheader("Model Comparison Matrix (Profit Regression)")
    if reg_metrics:
        reg_rows = []
        for model_name, metrics in reg_metrics.items():
            if isinstance(metrics, dict) and "Test R2" in metrics:
                reg_rows.append(
                    {
                        "Algorithm": model_name,
                        "CV R²": f"{metrics.get('Best CV R2', 0):.4f}",
                        "Test MAE": f"${metrics.get('Test MAE', 0):,.2f}",
                        "Test RMSE": f"${metrics.get('Test RMSE', 0):,.2f}",
                        "Test R²": f"{metrics.get('Test R2', 0):.4f}",
                    }
                )
        if reg_rows:
            st.dataframe(pd.DataFrame(reg_rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: SECTION 3 - BUSINESS INSIGHTS & EXPLAINABLE AI (XAI)
# ------------------------------------------------------------------------------
with tab4:
    st.header("🔍 Section 3: Explainable AI & Executive Insights")
    st.markdown("Rigorous model interpretability powered by game-theoretic SHAP and Gini feature importances.")

    # Executive Answers to Mandated Core Business Questions
    st.subheader("💡 Strategic Answers to Core Business Questions")

    q_answers = {
        "1. What drives returns?": (
            "Returns are primarily catalyzed by poor customer rating (<3.5), deep discounting (>=20%), "
            "and delayed fulfillment (>5 days). Concurrently delayed and heavily discounted orders exhibit an alarming 44%+ return rate."
        ),
        "2. What drives profit?": (
            "Profit is overwhelmingly dictated by Net Realized Sales (Effective Sales volume) and Product Category mix. "
            "Furniture orders yield the highest margin percentage (~16.2%), while Fashion and Electronics yield compressed margins (~2.8%-3.5%)."
        ),
        "3. How important is discounting?": (
            "Discounting is the #1 profit margin destroyer. Every percentage point of promotional discount directly reduces operating contribution. "
            "Simultaneously, excessive discounts stimulate impulse buying, which spikes post-purchase remorse and return volume."
        ),
        "4. How important is delivery delay?": (
            "Fulfillment latency is a critical operational risk. Shipments taking longer than 5 days experience a 41.2% return rate "
            "(vs. 23.1% for orders delivered within 5 days). Late shipments directly depress customer satisfaction ratings."
        ),
        "5. Which categories contribute most to risk & margins?": (
            "• **Highest Margin:** Furniture (16.2% margin) and Home Appliances (7.8% margin).\n"
            "• **Highest Operational Risk:** Fashion (high expectation variance) and Electronics (delay sensitivity)."
        ),
        "6. Which regions contribute most to profitability?": (
            "North and West regions lead profitability with margins hovering near ~8.9%. "
            "The South region generates high gross transaction volume but suffers the lowest net margins (4.76%) due to heavy discount reliance."
        ),
    }

    for question, answer in q_answers.items():
        with st.expander(f"📌 {question}", expanded=False):
            st.write(answer)

    st.markdown("---")
    st.subheader("📊 Visual Model Interpretability")

    col_x1, col_x2 = st.columns(2)

    with col_x1:
        st.markdown("#### Return Risk SHAP Summary")
        if (FIGURES_DIR / "return_shap_summary.png").exists():
            st.image(str(FIGURES_DIR / "return_shap_summary.png"), caption="SHAP Value Distribution - Return Model (XGBoost)")
        elif (FIGURES_DIR / "return_feature_importance.png").exists():
            st.image(str(FIGURES_DIR / "return_feature_importance.png"), caption="Gini Feature Importances - Return Model")

    with col_x2:
        st.markdown("#### Profit Regression SHAP Summary")
        if (FIGURES_DIR / "profit_shap_summary.png").exists():
            st.image(str(FIGURES_DIR / "profit_shap_summary.png"), caption="SHAP Value Distribution - Profit Regressor (Gradient Boosting)")
        elif (FIGURES_DIR / "profit_feature_importance.png").exists():
            st.image(str(FIGURES_DIR / "profit_feature_importance.png"), caption="Gini Feature Importances - Profit Regressor")

    st.markdown("---")
    st.subheader("🚀 Strategic Recommendations Panel")
    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.markdown(
            """
            **1. Discount Guardrails**
            - Cap promotional discounting at 18% for Fashion and Electronics.
            - Cease blanket 30%+ clearance markdowns in the South region.
            - Introduce volume-based discounts instead of flat order cuts.
            """
        )

    with rec_col2:
        st.markdown(
            """
            **2. 5-Day Delivery SLA**
            - Establish a regional fulfillment hub in the South to cut transit times.
            - Prioritize warehouse picking for high-ticket Furniture orders.
            - Implement automated tracking notifications when transit exceeds 3 days.
            """
        )

    with rec_col3:
        st.markdown(
            """
            **3. Pre-Fulfillment Interventions**
            - Intercept orders flagged as **High Return Risk (>60%)** before dispatch.
            - Provide automated customer address & sizing verification.
            - Dynamically adjust return shipping policies on deeply discounted units.
            """
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "NovaCart Solutions Pvt. Ltd. | Machine Learning & Business Intelligence Portfolio System | Powered by Scikit-Learn, XGBoost, SHAP & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
