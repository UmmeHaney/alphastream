import pandas as pd
import numpy as np

class TimeSeriesProcessor:
    def __init__(self, data_source, target_col: str):
        self.data_source = data_source
        self.target_col = target_col
        self.df = None

    def process(self, lags=[1, 2, 3, 7], windows=[3, 7]) -> pd.DataFrame:
        """Executes the data engineering pipeline sequentially."""
        self._ingest_and_clean()
        self._engineer_features(lags, windows)
        return self.df

    def _ingest_and_clean(self):
        """Loads data, infers datetime, and handles missing values."""
        self.df = pd.read_csv(self.data_source)
        
        # Heuristic datetime parsing
        datetime_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols:
            date_col = datetime_cols[0]
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df.set_index(date_col, inplace=True)
            self.df.sort_index(inplace=True)
        else:
            # Fallback if no explicit datetime column exists
            self.df.index = pd.date_range(start='2020-01-01', periods=len(self.df), freq='D')
            
        # Strict forward-fill to prevent look-ahead bias
        self.df = self.df.ffill().dropna()
        
        # Ensure target is numeric
        self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors='coerce')
        self.df = self.df.dropna(subset=[self.target_col])

    def _engineer_features(self, lags, windows):
        """Generates autoregressive lags and rolling statistics."""
        # Autoregressive lags
        for lag in lags:
            self.df[f'{self.target_col}_lag_{lag}'] = self.df[self.target_col].shift(lag)
            
        # Rolling statistics
        for window in windows:
            self.df[f'{self.target_col}_roll_mean_{window}'] = self.df[self.target_col].shift(1).rolling(window=window).mean()
            self.df[f'{self.target_col}_roll_std_{window}'] = self.df[self.target_col].shift(1).rolling(window=window).std()
            
        # Drop rows with NaNs introduced by shifting
        self.df.dropna(inplace=True)