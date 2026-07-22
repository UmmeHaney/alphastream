import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, model, max_samples=1000):
        self.model = model
        self.max_samples = max_samples
        self.sampled_indices = None
        self.expected_value = None

    def compute_shap_values(self, X: pd.DataFrame):
        """Computes SHAP values with uniform subsampling for memory safety."""
        if len(X) > self.max_samples:
            sampled_X = X.sample(n=self.max_samples, random_state=42).sort_index()
        else:
            sampled_X = X.copy()
            
        self.sampled_indices = sampled_X.index
        
        explainer = shap.TreeExplainer(self.model)
        self.expected_value = explainer.expected_value
        
        shap_values = explainer(sampled_X)
        return shap_values

    def get_global_importance(self, shap_values, feature_names: list) -> pd.DataFrame:
        """Aggregates absolute SHAP values for global feature ranking."""
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_abs_shap
        })
        return df.sort_values(by='Importance', ascending=False)

    def plot_waterfall(self, shap_values, integer_index: int):
        """Renders a localized waterfall chart for a specific indexed instance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[integer_index], show=False)
        plt.tight_layout()
        return fig

    def generate_executive_summary(self, shap_values, integer_index: int, feature_names: list, top_n: int = 3) -> str:
        """Produces a quick heuristic text summary of local feature drivers."""
        instance_shap = shap_values.values[integer_index]
        
        # Sort features by absolute impact magnitude
        sorted_indices = np.argsort(np.abs(instance_shap))[::-1]
        
        summary = f"Base Value: {shap_values.base_values[integer_index]:.2f}. \n\nTop {top_n} drivers for this specific prediction:\n"
        for i in range(top_n):
            idx = sorted_indices[i]
            feat_name = feature_names[idx]
            impact = instance_shap[idx]
            direction = "Increased" if impact > 0 else "Decreased"
            summary += f"- {feat_name}: {direction} prediction by {abs(impact):.2f}\n"
            
        return summary