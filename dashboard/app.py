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
    q_lower_cal = cal_preds['q_0.025'].values
    q_upper_cal = cal_preds['q_0.975'].values
    q_lower_test = test_preds['q_0.025'].values
    q_upper_test = test_preds['q_0.975'].values
    cqr = CQR(level=0.95)
    cqr.calibrate(y_cal, q_lower_cal, q_upper_cal)
    aci = AdaptiveConformalInference(level=0.95, gamma=0.01)
    aci_lower, aci_upper = aci.run_online(y_test, q_lower_test, q_upper_test, cqr.cal_scores)
    rel_widths = relative_interval_width(aci_lower, aci_upper, point_pred_test)
    tier_clf = RiskTierClassifier(method='percentile')
    tier_clf.fit(rel_widths)
    tiers = tier_clf.classify(rel_widths)
    results_df = test_df[['item_id', 'date', 'sales', 'is_volatile']].copy()
    results_df['point_forecast'] = point_pred_test
    results_df['lower_bound'] = aci_lower
    results_df['upper_bound'] = aci_upper
    results_df['interval_width'] = aci_upper - aci_lower
    results_df['relative_width'] = rel_widths
    results_df['risk_tier'] = tiers
    results_df['covered'] = ((y_test >= aci_lower) & (y_test <= aci_upper)).astype(int)
    overall_coverage = empirical_coverage(y_test, aci_lower, aci_upper)
    vol_mask = test_df['is_volatile'].values == 1
    calm_mask = ~vol_mask
    vol_coverage = empirical_coverage(y_test[vol_mask], aci_lower[vol_mask], aci_upper[vol_mask])
    calm_coverage = empirical_coverage(y_test[calm_mask], aci_lower[calm_mask], aci_upper[calm_mask])
    metrics = {'overall_coverage': overall_coverage, 'volatile_coverage': vol_coverage,
               'calm_coverage': calm_coverage, 'mean_width': mean_interval_width(aci_lower, aci_upper)}
    return results_df, metrics, aci

with st.spinner("Loading data and training models..."):
    results_df, metrics, aci = load_and_process_data()

st.title("🎯 Kizuna RiskTriage")
st.markdown("### Calibrated Uncertainty Quantification for Supply Chain Risk Triage")
st.markdown("*Making demand forecasts honest about what they don't know — and actionable for managers.*")
st.divider()


col1, col2, col3, col4 = st.columns(4)
with col1:
    cov_delta = metrics['overall_coverage'] - 0.95
    st.metric("Overall Coverage", f"{metrics['overall_coverage']:.1%}", delta=f"{cov_delta * 100:.1f}% from target", help="The % of time actual demand fell within the predicted range.")
with col2:
    vol_delta = metrics['volatile_coverage'] - 0.95
    st.metric("Volatile Coverage", f"{metrics['volatile_coverage']:.1%}", delta=f"{vol_delta * 100:.1f}% from target", help="Coverage calculated only for items with highly variable demand.")
with col3:
    high_pct = 100 * np.mean(results_df['risk_tier'] == 'High')
    st.metric("High Risk Items", f"{high_pct:.1f}%", delta="Need attention", help="% of catalog with highly uncertain forecasts requiring review.")
with col4:
    st.metric("Mean Interval Width", f"{metrics['mean_width']:.1f} units", help="Average gap between lower and upper bounds. Lower means more precision.")
st.divider()

st.sidebar.header("Controls")
selected_item = st.sidebar.selectbox("Select Product", results_df['item_id'].unique())
show_days = st.sidebar.slider("Days to display", 14, 120, 60)
st.sidebar.divider()
st.sidebar.markdown("### Risk Tier Actions")
st.sidebar.markdown("🟢 **Low**: Auto-replenish")
st.sidebar.markdown("🟡 **Medium**: Raise safety stock")
st.sidebar.markdown("🔴 **High**: Human review")

item_data = results_df[results_df['item_id'] == selected_item].tail(show_days).reset_index(drop=True)
st.subheader(f"📈 Demand Forecast: {selected_item} — last {len(item_data)} days")
if len(item_data) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['upper_bound'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['lower_bound'], mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(41,128,185,0.2)', name='95% Calibrated Interval'))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['sales'], mode='lines+markers', name='Actual Demand',
                             line=dict(color='black', width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=item_data.index, y=item_data['point_forecast'], mode='lines', name='Point Forecast',
                             line=dict(color='blue', width=1, dash='dash')))
    high_mask = item_data['risk_tier'] == 'High'
    if high_mask.any():
        fig.add_trace(go.Scatter(x=item_data.index[high_mask], y=item_data['sales'][high_mask], mode='markers',
                                 name='High Risk Day', marker=dict(color='red', size=10, symbol='diamond')))
    fig.update_layout(height=400, xaxis_title='Day', yaxis_title='Demand (units)',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0), margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True, key=f"forecast_{selected_item}_{show_days}")
    latest = item_data.iloc[-1]
    tier_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
    tier_action = {'Low': 'Auto-replenish', 'Medium': 'Raise safety stock', 'High': 'ESCALATE: Human review'}
    col_a, col_b, col_c = st.columns(3)
    with col_a: st.markdown(f"**Risk Tier:** {tier_color[latest['risk_tier']]} {latest['risk_tier']}")
    with col_b: st.markdown(f"**Forecast:** {latest['point_forecast']:.0f} [{latest['lower_bound']:.0f}—{latest['upper_bound']:.0f}]")
    with col_c: st.markdown(f"**Action:** {tier_action[latest['risk_tier']]}")

    # Per-product summary (makes the selection visibly responsive)
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.metric("Avg Demand (this product)", f"{item_data['sales'].mean():.1f}")
    with pcol2:
        st.metric("Avg Interval Width", f"{item_data['interval_width'].mean():.1f}")
    with pcol3:
        high_share = 100 * (item_data['risk_tier'] == 'High').mean()
        st.metric("% High-Risk Days", f"{high_share:.0f}%")
st.divider()

st.subheader("🚦 Risk Tier Overview")
col1, col2 = st.columns(2)
with col1:
    tier_counts_df = results_df.groupby('risk_tier').size().reset_index(name='count')
    fig_pie = px.pie(tier_counts_df, values='count', names='risk_tier',
                     color='risk_tier', color_discrete_map={'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'})
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    tier_stats = results_df.groupby('risk_tier').agg(stockout_rate=('covered', lambda x: 1-x.mean())).reset_index()
    fig_bar = px.bar(tier_stats, x='risk_tier', y='stockout_rate', color='risk_tier',
                     color_discrete_map={'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'},
                     title='Stockout Rate by Tier (Validation)')
    st.plotly_chart(fig_bar, use_container_width=True)
st.divider()

st.subheader("📊 Calibration Evidence")
col1, col2 = st.columns(2)
with col1:
    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(x=['Overall','Calm','Volatile'],
                             y=[metrics['overall_coverage'], metrics['calm_coverage'], metrics['volatile_coverage']],
                             marker_color=['steelblue','forestgreen','firebrick']))
    fig_cov.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Target 95%")
    fig_cov.update_layout(yaxis_range=[0.7,1.0], height=350, title='Coverage by Regime')
    st.plotly_chart(fig_cov, use_container_width=True)
with col2:
    aci_diag = pd.DataFrame({'Step': range(len(aci.coverage_history)),
                             'Running Coverage': pd.Series(aci.coverage_history).expanding().mean()})
    fig_aci = px.line(aci_diag, x='Step', y='Running Coverage', title='ACI Self-Correction')
    fig_aci.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target")
    fig_aci.update_layout(height=350)
    st.plotly_chart(fig_aci, use_container_width=True)

st.divider()
st.markdown("**Kizuna RiskTriage** | Team Kizuna | AI for Public Good Hackathon 2026")
