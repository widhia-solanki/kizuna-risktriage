"""
Kizuna RiskTriage — Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import create_synthetic_m5_like_data, train_test_split_temporal
from src.features import build_feature_matrix, get_feature_columns
from src.models import QuantileForecaster, PointForecaster
from src.calibration import CQR, AdaptiveConformalInference
from src.metrics import empirical_coverage, mean_interval_width, relative_interval_width
from src.risk_triage import RiskTierClassifier

st.set_page_config(page_title="Kizuna RiskTriage", page_icon="🎯", layout="wide")

# Hides the Plotly toolbar (modebar) that was overlapping the charts
PLOTLY_CONFIG = {"displayModeBar": False}


@st.cache_data
def load_and_process_data():
    df = create_synthetic_m5_like_data(n_items=30, n_days=800, seed=42)
    df = build_feature_matrix(df)
    feature_cols = get_feature_columns(df)
    train_df, cal_df, test_df = train_test_split_temporal(df, test_days=120, cal_days=60)
    X_train, y_train = train_df[feature_cols].values, train_df['sales'].values
    X_cal, y_cal = cal_df[feature_cols].values, cal_df['sales'].values
    X_test, y_test = test_df[feature_cols].values, test_df['sales'].values
    quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
    qf = QuantileForecaster(quantiles=quantiles)
    qf.fit(X_train, y_train)
    pf = PointForecaster()
    pf.fit(X_train, y_train)
    cal_preds = qf.predict(X_cal)
    test_preds = qf.predict(X_test)
    point_pred_test = np.maximum(qf.models[0.5].predict(X_test), 0)
    q_lower_test = test_preds['q_0.025'].values
    q_upper_test = test_preds['q_0.975'].values
    cqr = CQR(level=0.95)
    cqr.calibrate(y_cal, cal_preds['q_0.025'].values, cal_preds['q_0.975'].values)
    aci = AdaptiveConformalInference(level=0.95, gamma=0.01)
    aci_lower, aci_upper = aci.run_online(y_test, q_lower_test, q_upper_test, cqr.cal_scores)
    return _assemble(test_df, y_test, point_pred_test, aci_lower, aci_upper,
                     q_lower_test, q_upper_test, aci)



def _assemble(test_df, y_test, point_pred_test, aci_lower, aci_upper,
              q_lower_test, q_upper_test, aci):
    rel_widths = relative_interval_width(aci_lower, aci_upper, point_pred_test)
    tier_clf = RiskTierClassifier(method='percentile')
    tier_clf.fit(rel_widths)
    tiers = tier_clf.classify(rel_widths)
    results_df = test_df[['item_id', 'date', 'sales', 'is_volatile']].copy()
    results_df['point_forecast'] = point_pred_test
    results_df['lower_bound'] = aci_lower
    results_df['upper_bound'] = aci_upper
    results_df['raw_lower'] = q_lower_test
    results_df['raw_upper'] = q_upper_test
    results_df['interval_width'] = aci_upper - aci_lower
    results_df['relative_width'] = rel_widths
    results_df['risk_tier'] = tiers
    results_df['covered'] = ((y_test >= aci_lower) & (y_test <= aci_upper)).astype(int)
    vol_mask = test_df['is_volatile'].values == 1
    calm_mask = ~vol_mask
    metrics = {
        'overall_coverage': empirical_coverage(y_test, aci_lower, aci_upper),
        'volatile_coverage': empirical_coverage(y_test[vol_mask], aci_lower[vol_mask], aci_upper[vol_mask]),
        'calm_coverage': empirical_coverage(y_test[calm_mask], aci_lower[calm_mask], aci_upper[calm_mask]),
        'mean_width': mean_interval_width(aci_lower, aci_upper),
    }
    return results_df, metrics, aci


with st.spinner("Loading data and training models..."):
    results_df, metrics, aci = load_and_process_data()

st.title("🎯 Kizuna RiskTriage")
st.markdown("### Calibrated Uncertainty Quantification for Supply Chain Risk Triage")
st.caption("Making demand forecasts honest about what they don't know — and actionable for managers.")
st.divider()
# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
selected_item = st.sidebar.selectbox("Select Product", sorted(results_df['item_id'].unique()))
show_days = st.sidebar.slider("Days to display", 14, 120, 60)
show_raw_qr = st.sidebar.checkbox(
    "Show Raw QR (uncalibrated) band", value=False,
    help="Overlays the ORIGINAL uncalibrated quantile-regression band (red, dotted) on top of "
         "our calibrated band (blue). Use it to SEE how the raw model's interval is too narrow "
         "and under-covers — that is the problem our calibration fixes.")
st.sidebar.divider()
st.sidebar.markdown("### Risk Tier Actions")
st.sidebar.markdown("🟢 **Low** → Auto-replenish")
st.sidebar.markdown("🟡 **Medium** → Raise safety stock")
st.sidebar.markdown("🔴 **High** → Human review")

# ---------------- Product-Specific Metrics (Top of Page) ----------------
item_all = results_df[results_df['item_id'] == selected_item]
item_cov = item_all['covered'].mean()
item_width = item_all['interval_width'].mean()
item_high_risk = (item_all['risk_tier'] == 'High').mean()
item_sales = item_all['sales'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Item Coverage", f"{item_cov:.1%}", f"{item_cov-0.95:+.1%} vs target", help="The % of time actual demand fell within the predicted range for this specific item.")
col2.metric("Avg Demand", f"{item_sales:.1f} units", help="Average daily demand for this item.")
col3.metric("High-Risk Days", f"{100*item_high_risk:.1f}%", help="% of days this item had highly uncertain forecasts requiring human review.")
col4.metric("Mean Interval Width", f"{item_width:.1f} units", help="Average gap between lower and upper bounds for this item. Lower means more precision.")
st.divider()

# ---------------- Per-product forecast (CHANGES per product) ----------------
item_data = results_df[results_df['item_id'] == selected_item].tail(show_days).reset_index(drop=True)
st.subheader(f"📈 Demand Forecast: {selected_item} — last {len(item_data)} days")
st.caption("This section updates every time you change the selected product.")
if len(item_data) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['upper_bound'], mode='lines',
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['lower_bound'], mode='lines',
                             line=dict(width=0), fill='tonexty', fillcolor='rgba(41,128,185,0.2)',
                             name='95% Calibrated (CQR+ACI)'))
    if show_raw_qr:
        fig.add_trace(go.Scatter(x=item_data.index, y=item_data['raw_upper'], mode='lines',
                                 line=dict(color='red', width=1, dash='dot'), name='Raw QR upper (uncalibrated)'))
        fig.add_trace(go.Scatter(x=item_data.index, y=item_data['raw_lower'], mode='lines',
                                 line=dict(color='red', width=1, dash='dot'), name='Raw QR lower (uncalibrated)'))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['sales'], mode='lines+markers',
                             name='Actual Demand', line=dict(color='black', width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['point_forecast'], mode='lines',
                             name='Point Forecast', line=dict(color='blue', width=1, dash='dash')))
    high_mask = item_data['risk_tier'] == 'High'
    if high_mask.any():
        fig.add_trace(go.Scatter(x=item_data.index[high_mask], y=item_data['sales'][high_mask],
                                 mode='markers', name='High Risk Day',
                                 marker=dict(color='red', size=10, symbol='diamond')))
    fig.update_layout(height=420, xaxis_title='Day', yaxis_title='Demand (units)',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0), margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True,
                    key=f"forecast_{selected_item}_{show_days}_{show_raw_qr}", config=PLOTLY_CONFIG)
    latest = item_data.iloc[-1]
    tier = latest['risk_tier']
    action = {'Low': 'Auto-replenish (nominal safety stock)',
              'Medium': 'Raise safety stock to calibrated quantile; dashboard flag',
              'High': 'ESCALATE: Human review required; consider dual-sourcing'}[tier]
    
    tier_icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}[tier]
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.5])
        c1.metric("Latest Risk Tier", f"{tier_icon} {tier}")
        c2.metric("Point Forecast", f"{latest['point_forecast']:.0f}", f"{latest['lower_bound']:.0f} to {latest['upper_bound']:.0f} units", delta_color="off")
        with c3:
            if tier == 'Low': st.success(f"**Action:** {action}")
            elif tier == 'Medium': st.warning(f"**Action:** {action}")
            else: st.error(f"**Action:** {action}")


st.divider()

# ---------------- Portfolio overview (does NOT change per product) ----------------
st.subheader("🚦 Portfolio KPIs — All Products")
st.caption("System-level metrics summarizing the health of your ENTIRE catalog.")

with st.container(border=True):
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    gcol1.metric("Overall Coverage", f"{metrics['overall_coverage']:.1%}", f"{metrics['overall_coverage']-0.95:+.1%} vs target", help="The % of time actual demand fell within the predicted range across all products.")
    gcol2.metric("Volatile Coverage", f"{metrics['volatile_coverage']:.1%}", f"{metrics['volatile_coverage']-0.95:+.1%} vs target", help="Coverage calculated only for items with highly variable demand.")
    gcol3.metric("High Risk Items", f"{100*np.mean(results_df['risk_tier']=='High'):.1f}%", delta="Need attention", help="% of catalog with highly uncertain forecasts requiring review.")
    gcol4.metric("Mean Interval Width", f"{metrics['mean_width']:.1f} units", help="Average gap between lower and upper bounds. Lower means more precision.")

st.write("")
st.subheader("🚦 Risk Tier Distribution")
st.caption("Portfolio-wide summary across the whole catalog. By design this does NOT change "
           "when you switch a single product.")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        tier_counts_df = results_df.groupby('risk_tier').size().reset_index(name='count')
        fig_pie = px.pie(tier_counts_df, values='count', names='risk_tier', color='risk_tier',
                         color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
                         title='Risk Tier Distribution')
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)
with col2:
    with st.container(border=True):
        tier_stats = results_df.groupby('risk_tier').agg(
            stockout_rate=('covered', lambda x: 1 - x.mean())).reset_index()
        fig_bar = px.bar(tier_stats, x='risk_tier', y='stockout_rate', color='risk_tier',
                         color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
                         title='Stockout Rate by Tier (Validation)')
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)
st.divider()

# ---------------- Calibration evidence (whole-system, constant) ----------------
st.subheader("📊 Calibration Evidence — Whole System")
st.caption("System-level proof that our uncertainty is honest. Also constant across products by design.")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(x=['Overall', 'Calm', 'Volatile'],
                                 y=[metrics['overall_coverage'], metrics['calm_coverage'], metrics['volatile_coverage']],
                                 marker_color=['steelblue', 'forestgreen', 'firebrick']))
        fig_cov.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Target 95%")
        fig_cov.update_layout(yaxis_range=[0.7, 1.0], height=350, title='Coverage by Regime')
        st.plotly_chart(fig_cov, use_container_width=True, config=PLOTLY_CONFIG)
with col2:
    with st.container(border=True):
        aci_diag = pd.DataFrame({'Step': range(len(aci.coverage_history)),
                                 'Running Coverage': pd.Series(aci.coverage_history).expanding().mean()})
        fig_aci = px.line(aci_diag, x='Step', y='Running Coverage', title='ACI Self-Correction Over Time')
        fig_aci.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target")
        fig_aci.update_layout(height=350)
        st.plotly_chart(fig_aci, use_container_width=True, config=PLOTLY_CONFIG)

st.divider()
st.markdown("**Kizuna RiskTriage** | Team Kizuna | AI for Public Good Hackathon 2026")
