import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class DashboardVisualizer:
    def render_metric_cards(self, metrics: dict):
        """Renders high-level KPI cards for model evaluation."""
        cols = st.columns(len(metrics))
        for col, (metric_name, value) in zip(cols, metrics.items()):
            if metric_name in ['MAE', 'RMSE']:
                formatted_value = f"{value:.4f}"
            elif metric_name == 'MAPE':
                formatted_value = f"{value:.2f}%"
            else:
                formatted_value = f"{value:.4f}"
            
            col.metric(label=metric_name, value=formatted_value)

    def plot_forecast_with_intervals(self, df: pd.DataFrame, target_col: str, title: str = "Forecast Overview"):
        """Renders an interactive Plotly chart with historical actuals and future forecast bounds."""
        fig = go.Figure()

        # Historical Actuals Trace
        if target_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[target_col],
                mode='lines',
                name='Actuals',
                line=dict(color='#2E86C1', width=2)
            ))

        # Predicted Future Trace
        if 'prediction' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['prediction'],
                mode='lines',
                name='Forecast',
                line=dict(color='#E74C3C', width=2, dash='dash')
            ))

        # Confidence Intervals
        if 'upper_bound' in df.columns and 'lower_bound' in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
                y=pd.concat([df['upper_bound'], df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='95% Confidence Interval'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Target Value",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """Renders a horizontal bar chart of global SHAP feature importance."""
        fig = px.bar(
            importance_df.head(10), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Global Feature Drivers (SHAP)',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template="plotly_white"
        )
        return fig