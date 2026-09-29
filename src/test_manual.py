import os
import joblib
import torch
import numpy as np
import pandas as pd

from dataset import FEATURE_COLS, TARGET_COL
from model import DieselPINN

def load_artifacts():
    weights_path = os.path.join("models", "diesel_pinn_weights.pt")
    scaler_path = os.path.join("models", "diesel_scaler.pkl")

    if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model artifacts not found. Run `python src/train.py` first.")

    scaler = joblib.load(scaler_path)
    model = DieselPINN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

def predict_blend(model, scaler, rpm, load, fuel_flow, tau_ms):
    input_vector = np.array([[rpm, load, fuel_flow, tau_ms]])
    scaled_vector = scaler.transform(input_vector)
    
    with torch.no_grad():
        pred = model(torch.tensor(scaled_vector, dtype=torch.float32)).item()
    
    return max(0.0, float(pred))

def main():
    model, scaler = load_artifacts()
    csv_path = os.path.join("data", "diesel_isobutanol_augmented_dataset.csv")
    df = pd.read_csv(csv_path)

    print("=" * 65)
    print("      DIESEL-ISOBUTANOL PINN: MANUAL INFERENCE TESTING       ")
    print("=" * 65)
    print(f"Dataset loaded: {len(df)} total rows.")
    print("Select an option:")
    print("  [1] Test an exact row from the dataset (compare actual vs predicted)")
    print("  [2] Enter custom manual telemetry values")
    print("  [3] Exit")
    print("=" * 65)

    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()

        if choice == '1':
            try:
                row_idx = int(input(f"Enter row index (0 to {len(df)-1}): ").strip())
                if not (0 <= row_idx < len(df)):
                    print(f"Error: Index must be between 0 and {len(df)-1}.")
                    continue
                
                row = df.iloc[row_idx]
                rpm = float(row['engine_rpm'])
                load = float(row['engine_load'])
                fuel_flow = float(row['fuel_flow_gps'])
                tau_ms = float(row['ignition_delay_est_ms'])
                actual_blend = float(row[TARGET_COL])

                pred_blend = predict_blend(model, scaler, rpm, load, fuel_flow, tau_ms)
                error = abs(actual_blend - pred_blend)

                print("-" * 55)
                print(f"Dataset Row #{row_idx} Telemetry:")
                print(f"  • Engine RPM:               {rpm:.0f} RPM")
                print(f"  • Engine Load:              {load:.4f}")
                print(f"  • Fuel Flow:                {fuel_flow:.4f} g/s")
                print(f"  • Ignition Delay:           {tau_ms:.4f} ms")
                print("-" * 55)
                print(f"  * Actual Isobutanol Blend:   {actual_blend:.2f}%")
                print(f"  * PINN Predicted Blend:     {pred_blend:.2f}%")
                print(f"  * Absolute Error:           {error:.4f}%")
                print("-" * 55)

            except ValueError:
                print("Invalid input! Please enter a valid integer index.")

        elif choice == '2':
            try:
                rpm = float(input("Enter Engine RPM (e.g. 2000): ").strip())
                load = float(input("Enter Engine Load (e.g. 0.50): ").strip())
                fuel_flow = float(input("Enter Fuel Flow in g/s (e.g. 1.25): ").strip())
                tau_ms = float(input("Enter Ignition Delay in ms (e.g. 1.85): ").strip())

                pred_blend = predict_blend(model, scaler, rpm, load, fuel_flow, tau_ms)

                print("-" * 55)
                print("Manual Inputs:")
                print(f"  RPM={rpm}, Load={load}, FuelFlow={fuel_flow} g/s, IgnitionDelay={tau_ms} ms")
                print(f"  -> Predicted Isobutanol Blend: {pred_blend:.2f}%")
                print("-" * 55)

            except ValueError:
                print("Invalid numerical value. Please re-enter.")

        elif choice == '3':
            print("Exiting test session.")
            break
        else:
            print("Invalid option. Enter 1, 2, or 3.")

if __name__ == "__main__":
    main()