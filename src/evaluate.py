import os
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import FEATURE_COLS, TARGET_COL
from model import DieselPINN

def evaluate():
    # 1. Load data & artifacts
    df = pd.read_csv("data/diesel_isobutanol_augmented_dataset.csv")
    scaler = joblib.load("models/diesel_scaler.pkl")
    
    model = DieselPINN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load("models/diesel_pinn_weights.pt"))
    model.eval()

    # 2. Predict on full dataset
    X_raw = df[FEATURE_COLS].values
    y_true = df[TARGET_COL].values
    
    X_scaled = scaler.transform(X_raw)
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()

    # 3. Compute Metrics
    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    print("=" * 45)
    print("      DIESEL-ISOBUTANOL PINN EVALUATION      ")
    print("=" * 45)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}% isobutanol")
    print(f"R² Score:                  {r2:.4f}")
    print("=" * 45)

    # 4. Generate Parity Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, preds, alpha=0.35, color='#1f77b4', edgecolors='none', label='Test Data Points')
    max_val = max(y_true.max(), preds.max()) + 1.0
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Parity (y = x)')
    plt.title("Ground Truth vs Predicted Isobutanol % (PINN)")
    plt.xlabel("Actual Isobutanol Blend (%)")
    plt.ylabel("PINN Predicted Isobutanol Blend (%)")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = "models/parity_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Parity plot saved to: {plot_path}")

if __name__ == "__main__":
    evaluate()