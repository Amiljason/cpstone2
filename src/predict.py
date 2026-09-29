import os
import joblib
import torch
import pandas as pd
import numpy as np

from dataset import FEATURE_COLS, TARGET_COL
from model import DieselPINN

def load_inference_artifacts():
    scaler_path = os.path.join("models", "diesel_scaler.pkl")
    weights_path = os.path.join("models", "diesel_pinn_weights.pt")

    if not os.path.exists(scaler_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("Model artifacts missing. Run `python src/train.py` first.")

    scaler = joblib.load(scaler_path)
    model = DieselPINN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

def predict_single(engine_rpm, engine_load, fuel_flow_gps, ignition_delay_est_ms):
    model, scaler = load_inference_artifacts()
    
    raw_inputs = np.array([[engine_rpm, engine_load, fuel_flow_gps, ignition_delay_est_ms]])
    scaled_inputs = scaler.transform(raw_inputs)
    
    with torch.no_grad():
        prediction = model(torch.tensor(scaled_inputs, dtype=torch.float32)).item()
    
    return max(0.0, prediction)

def compare_random_samples(n_samples=5):
    model, scaler = load_inference_artifacts()
    df = pd.read_csv("data/diesel_isobutanol_augmented_dataset.csv")
    
    sample = df.sample(n_samples, random_state=42)
    X = sample[FEATURE_COLS].values
    y_actual = sample[TARGET_COL].values
    
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()
        preds = np.clip(preds, 0.0, None)

    comparison_df = sample[FEATURE_COLS].copy()
    comparison_df['Actual Blend (%)'] = np.round(y_actual, 2)
    comparison_df['Predicted Blend (%)'] = np.round(preds, 2)
    comparison_df['Error (%)'] = np.round(np.abs(y_actual - preds), 2)
    
    print("\n" + "=" * 80)
    print("           SAMPLE TEST: 4 INPUT TELEMETRY -> ACTUAL VS PREDICTED           ")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    # 1. Show side-by-side comparisons on test records
    compare_random_samples(n_samples=6)

    # 2. Example of manual inference call
    example_rpm = 2500
    example_load = 0.60
    example_fuel_flow = 2.45
    example_tau = 1.35
    
    pred = predict_single(example_rpm, example_load, example_fuel_flow, example_tau)
    print(f"\n[Manual Input Test]")
    print(f"Inputs: RPM={example_rpm}, Load={example_load}, FuelFlow={example_fuel_flow} g/s, Tau={example_tau} ms")
    print(f"-> Predicted Isobutanol Blend: {pred:.2f}%\n")