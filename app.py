import os
import time
import streamlit as st
import pandas as pd
import numpy as np

# Import our custom enterprise modules
from core.data_processor import TimeSeriesProcessor
from core.model_engine import ForecastingEngine
from core.explainer import ModelExplainer
from core.llm_insights import BusinessAnalyst
from utils.visualizer import DashboardVisualizer

# -----------------------------------------------------------------------------
# Configuration & Caching
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AlphaStream AI", page_icon="📈", layout="wide")

@st.cache_data(show_spinner=False)
def compute_cached_shap(_explainer, X, model_timestamp):
    """Caches SHAP array to RAM. Invalidates if X or model_timestamp changes."""
    return _explainer.compute_shap_values(X)

def invalidate_downstream_state(level="data"):
    """Strict state invalidation to prevent zombie tensors and UI crashes."""
    if level == "data":
        st.session_state.engine = None
        st.session_state.metrics = None
        st.session_state.model_timestamp = None
        level = "model" # cascade down
    if level == "model":
        st.session_state.explainer = None
        st.session_state.shap_values = None
        st.session_state.insights = None

def init_session_state():
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'target_col' not in st.session_state: st.session_state.target_col = None
    if 'engine' not in st.session_state: st.session_state.engine = None
    if 'model_timestamp' not in st.session_state: st.session_state.model_timestamp = None
    if 'metrics' not in st.session_state: st.session_state.metrics = None
    if 'explainer' not in st.session_state: st.session_state.explainer = None
    if 'shap_values' not in st.session_state: st.session_state.shap_values = None
    if 'insights' not in st.session_state: st.session_state.insights = None
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    if 'llm_model' not in st.session_state: st.session_state.llm_model = "gemini-2.5-flash"

init_session_state()
viz = DashboardVisualizer()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Train Model", "Forecast", "Explainability", "AI Business Analyst"])

st.sidebar.markdown("---")
st.session_state.api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
st.session_state.llm_model = st.sidebar.selectbox("GenAI Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page == "Upload Data":
    st.title("Data Ingestion")
    uploaded_file = st.file_uploader("Upload time-series CSV", type=["csv"])
    
    if uploaded_file:
        temp_df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select Target Column:", temp_df.columns)
        
        if st.button("Process Features"):
            with st.spinner("Engineering features..."):
                uploaded_file.seek(0)
                processor = TimeSeriesProcessor(data_source=uploaded_file, target_col=target_col)
                st.session_state.processed_data = processor.process()
                st.session_state.target_col = target_col
                invalidate_downstream_state("data")
            st.success("Pipeline executed.")
            st.dataframe(st.session_state.processed_data.head())

elif page == "Train Model":
    st.title("Model Training Engine")
    if st.session_state.processed_data is None:
        st.warning("Upload data first.")
    else:
        test_size = st.slider("Holdout Test Size (%)", 10, 40, 20) / 100.0
        
        if st.button("Train Model"):
            with st.spinner("Training XGBoost Regressor..."):
                engine = ForecastingEngine()
                st.session_state.metrics = engine.train_and_evaluate(
                    st.session_state.processed_data, 
                    st.session_state.target_col, 
                    test_size
                )
                st.session_state.engine = engine
                st.session_state.model_timestamp = time.time()
                invalidate_downstream_state("model")
                
        if st.session_state.metrics:
            viz.render_metric_cards(st.session_state.metrics)
            
        # V1.0 Critical Fix: Model Serialization UI Hooks
        st.markdown("---")
        st.subheader("Model Persistence")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model Artifacts", disabled=st.session_state.engine is None):
                try:
                    st.session_state.engine.save_model("saved_models/alphastream_xgboost")
                    st.success("Model securely persisted to disk.")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
                    
        with col2:
            if st.button("📂 Load Existing Model"):
                try:
                    engine = ForecastingEngine()
                    engine.load_model("saved_models/alphastream_xgboost")
                    st.session_state.engine = engine
                    st.session_state.model_timestamp = time.time()
                    invalidate_downstream_state("model")
                    st.success("Model hydrated securely from disk. Evaluation metrics cleared; proceed to Forecasting.")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

elif page == "Forecast":
    st.title("Recursive Forecasting")
    if not st.session_state.engine:
        st.warning("Train the model first.")
    else:
        steps = st.number_input("Forecast Steps", min_value=1, max_value=90, value=14)
        if st.button("Generate Forecast"):
            with st.spinner("Executing recursive autoregressive loop..."):
                results_df = st.session_state.engine.forecast_future(
                    df=st.session_state.processed_data,
                    target_col=st.session_state.target_col,
                    steps=steps,
                    lags=[1, 2, 3, 7],
                    windows=[3, 7]
                )
                
                # V1.0 Critical Fix: Forecast Visualization Detachment
                historical_tail = st.session_state.processed_data[[st.session_state.target_col]].tail(90).copy()
                
                last_hist_idx = historical_tail.index[-1]
                last_hist_val = float(historical_tail.loc[last_hist_idx, st.session_state.target_col])
                historical_tail.loc[last_hist_idx, 'prediction'] = last_hist_val
                historical_tail.loc[last_hist_idx, 'lower_bound'] = last_hist_val
                historical_tail.loc[last_hist_idx, 'upper_bound'] = last_hist_val
                
                combined_df = pd.concat([historical_tail, results_df])
                
                fig = viz.plot_forecast_with_intervals(
                    combined_df, 
                    st.session_state.target_col, 
                    title="AlphaStream Historical Context & Future Forecast"
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Explainability":
    st.title("Explainable AI (SHAP)")
    if not st.session_state.engine:
        st.warning("Train the model first.")
    else:
        df = st.session_state.processed_data
        target = st.session_state.target_col
        X = df.drop(columns=[target])
        
        if not st.session_state.explainer:
            st.session_state.explainer = ModelExplainer(st.session_state.engine.model)
            
        with st.spinner("Fetching SHAP Explanations..."):
            st.session_state.shap_values = compute_cached_shap(
                st.session_state.explainer, X, st.session_state.model_timestamp
            )
            
        explainer = st.session_state.explainer
        shap_vals = st.session_state.shap_values
        
        importance_df = explainer.get_global_importance(shap_vals, X.columns.tolist())
        fig_bar = viz.plot_feature_importance(importance_df)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Local Explanation")
        
        sampled_dates = explainer.sampled_indices
        selected_date = st.selectbox("Select Forecast Date to Analyze", sampled_dates.strftime("%Y-%m-%d %H:%M:%S"))
        
        date_obj = pd.to_datetime(selected_date)
        integer_idx = int(np.where(sampled_dates == date_obj)[0][0])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(explainer.generate_executive_summary(shap_vals, integer_idx, X.columns.tolist(), top_n=4))
        with col2:
            st.pyplot(explainer.plot_waterfall(shap_vals, integer_index=integer_idx))

elif page == "AI Business Analyst":
    st.title("AI Business Analyst")
    
    if not st.session_state.engine or not st.session_state.metrics:
        st.warning("Action Required: Train the XGBoost model first to generate evaluation metrics.")
    elif not st.session_state.shap_values:
        st.warning("Action Required: Run the Explainability module first to compute SHAP values.")
    elif not st.session_state.api_key:
        st.warning("Action Required: Enter your Gemini API key in the sidebar configuration panel.")
    else:
        if not st.session_state.insights:
            if st.button("Generate Strategy Briefing"):
                with st.spinner("Consulting the Gemini API for strategic synthesis..."):
                    try:
                        df = st.session_state.processed_data
                        target = st.session_state.target_col
                        X = df.drop(columns=[target])
                        
                        preds = st.session_state.engine.model.predict(X)
                        forecast_summary = {
                            "mean_predicted_value": float(np.mean(preds)),
                            "max_predicted_value": float(np.max(preds)),
                            "min_predicted_value": float(np.min(preds))
                        }
                        
                        importance_df = st.session_state.explainer.get_global_importance(
                            st.session_state.shap_values, X.columns.tolist()
                        )
                        shap_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
                        
                        executive_context = f"Analyzing forecast target '{target}' using an XGBoost model. The goal is to maximize ROI and mitigate downside risk."
                        
                        analyst = BusinessAnalyst(
                            api_key=st.session_state.api_key,
                            model_name=st.session_state.llm_model
                        )
                        
                        st.session_state.insights = analyst.generate_insights(
                            forecast_summary=forecast_summary,
                            metrics=st.session_state.metrics,
                            shap_importance=shap_dict,
                            executive_context=executive_context
                        )
                        st.success("Strategic briefing generated successfully.")
                        
                    except Exception as e:
                        st.error(f"API Integration Failure: {e}")
                        
        if st.session_state.insights:
            ins = st.session_state.insights
            
            st.markdown("---")
            st.markdown("### 📊 Executive Summary")
            st.write(ins.get("executive_summary", "Data unavailable."))
            
            st.markdown("### 🎯 Confidence Assessment")
            st.write(ins.get("confidence_assessment", "Data unavailable."))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🚀 Key Drivers")
                for driver in ins.get("key_drivers", []):
                    st.markdown(f"- {driver}")
                
                st.markdown("### ⚠️ Risks")
                for risk in ins.get("risks", []):
                    st.markdown(f"- {risk}")
            
            with col2:
                st.markdown("### 💡 Opportunities")
                for opp in ins.get("opportunities", []):
                    st.markdown(f"- {opp}")
                    
                st.markdown("### ✅ Recommended Actions")
                for action in ins.get("recommended_actions", []):
                    st.markdown(f"- {action}")