import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Strictly the 4 core physical telemetry inputs
FEATURE_COLS = [
    'engine_rpm', 
    'engine_load', 
    'fuel_flow_gps', 
    'ignition_delay_est_ms'
]

TARGET_COL = 'isobutanol_pct'

class DieselDataset(Dataset):
    def __init__(self, df, scaler=None, is_train=True):
        self.raw_data = df.reset_index(drop=True)
        
        X = self.raw_data[FEATURE_COLS].values
        y = self.raw_data[TARGET_COL].values.reshape(-1, 1)
        
        if is_train:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            self.X_scaled = self.scaler.transform(X)
            
        self.y = y

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X_scaled[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32),
            'engine_load': torch.tensor(self.raw_data.at[idx, 'engine_load'], dtype=torch.float32),
            'fuel_flow_gps': torch.tensor(self.raw_data.at[idx, 'fuel_flow_gps'], dtype=torch.float32),
            'ignition_delay_est_ms': torch.tensor(self.raw_data.at[idx, 'ignition_delay_est_ms'], dtype=torch.float32)
        }