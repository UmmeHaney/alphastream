import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ForecastingEngine:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        self.is_trained = False
        self.residual_std = 0.0

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> dict:
        """Executes chronological train/test split and calculates out-of-sample metrics."""
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        self.model.fit(X_train, y_train)
        self.is_trained = True

        predictions = self.model.predict(X_test)
        
        # Calculate residual variance for confidence bounds
        residuals = y_test - predictions
        self.residual_std = float(np.std(residuals))

        return {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": float(r2_score(y_test, predictions)),
            "MAPE": float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)
        }

    def predict_with_intervals(self, X: pd.DataFrame, confidence_level: float = 1.96) -> pd.DataFrame:
        """Scores historical data with confidence bounds, enforcing exact schema validation."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        
        # V1.0 Hardened Schema Validation
        if hasattr(self.model, 'feature_names_in_'):
            expected = list(self.model.feature_names_in_)
            actual = list(X.columns)
            
            if actual != expected:
                raise ValueError(
                    f"Feature schema mismatch. The uploaded dataset columns and order do not match the model's training schema.\n"
                    f"Expected: {expected}\n"
                    f"Actual: {actual}"
                )
            
            non_numeric = [
                c for c in expected
                if not pd.api.types.is_numeric_dtype(X[c])
            ]
            
            if non_numeric:
                raise TypeError(
                    f"Non-numeric feature(s) detected in expected feature set: {non_numeric}"
                )
        
        predictions = self.model.predict(X)
        margin_of_error = confidence_level * self.residual_std
        
        return pd.DataFrame({
            'prediction': predictions,
            'lower_bound': predictions - margin_of_error,
            'upper_bound': predictions + margin_of_error
        }, index=X.index)

    def forecast_future(self, df: pd.DataFrame, target_col: str, steps: int, lags: list, windows: list) -> pd.DataFrame:
        """Executes a recursive out-of-sample forecast loop."""
        buffer_df = df.copy()
        
        # Infer future datetime index safely
        freq = pd.infer_freq(buffer_df.index)
        if not freq:
            delta = buffer_df.index.to_series().diff().mode()[0]
            future_dates = [buffer_df.index[-1] + (i * delta) for i in range(1, steps + 1)]
        else:
            future_dates = pd.date_range(start=buffer_df.index[-1], periods=steps + 1, freq=freq)[1:]

        future_predictions = []
        future_lower = []
        future_upper = []
        margin_of_error = 1.96 * self.residual_std

        for future_date in future_dates:
            # Construct feature vector for the next step
            next_row = {}
            for lag in lags:
                next_row[f'{target_col}_lag_{lag}'] = buffer_df.iloc[-lag][target_col]
            for window in windows:
                next_row[f'{target_col}_roll_mean_{window}'] = buffer_df.iloc[-window:][target_col].mean()
                next_row[f'{target_col}_roll_std_{window}'] = buffer_df.iloc[-window:][target_col].std()
                
            X_next = pd.DataFrame([next_row])
            
            # Reorder explicitly to match training schema
            if hasattr(self.model, 'feature_names_in_'):
                X_next = X_next[self.model.feature_names_in_]
                
            pred_val = self.model.predict(X_next)[0]
            
            future_predictions.append(pred_val)
            future_lower.append(pred_val - margin_of_error)
            future_upper.append(pred_val + margin_of_error)
            
            # Append to buffer for the next recursive step
            new_row = pd.DataFrame({target_col: [pred_val]}, index=[future_date])
            buffer_df = pd.concat([buffer_df, new_row])

        return pd.DataFrame({
            'prediction': future_predictions,
            'lower_bound': future_lower,
            'upper_bound': future_upper
        }, index=future_dates)

    def save_model(self, filepath_prefix: str):
        """Serializes the XGBoost model natively via JSON (Pickle-free)."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        self.model.save_model(f"{filepath_prefix}.json")
        
        import json
        metadata = {
            "residual_std": self.residual_std,
            "feature_names_in_": list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else None
        }
        with open(f"{filepath_prefix}_meta.json", "w") as f:
            json.dump(metadata, f)

    def load_model(self, filepath_prefix: str):
        """Hydrates the model and metadata securely from disk."""
        self.model.load_model(f"{filepath_prefix}.json")
        
        import json
        with open(f"{filepath_prefix}_meta.json", "r") as f:
            metadata = json.load(f)
            self.residual_std = metadata.get("residual_std", 0.0)
            features = metadata.get("feature_names_in_")
            if features:
                self.model.feature_names_in_ = np.array(features)
                
        self.is_trained = True