import os
import joblib
import torch
import numpy as np
import pandas as pd

from dataset import FEATURE_COLS, TARGET_COL
from model import DieselPINN
from engine_analytics import compute_engine_analytics

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

def run_diagnostics(row_idx=None, manual_inputs=None):
    model, scaler = load_artifacts()
    
    if manual_inputs is None:
        df = pd.read_csv("data/diesel_isobutanol_augmented_dataset.csv")
        idx = 2 if row_idx is None else row_idx
        row = df.iloc[idx]
        rpm = float(row['engine_rpm'])
        load = float(row['engine_load'])
        fuel_flow = float(row['fuel_flow_gps'])
        tau_ms = float(row['ignition_delay_est_ms'])
        actual_blend = float(row[TARGET_COL])
        title = f"DATASET ROW #{idx} (Ground Truth: {actual_blend:.2f}% Isobutanol)"
    else:
        rpm, load, fuel_flow, tau_ms = manual_inputs
        title = "MANUAL TELEMETRY INPUT"

    # 1. PINN Model Prediction
    x_raw = np.array([[rpm, load, fuel_flow, tau_ms]])
    x_scaled = scaler.transform(x_raw)
    with torch.no_grad():
        pred_blend = float(model(torch.tensor(x_scaled, dtype=torch.float32)).item())
    pred_blend = max(0.0, pred_blend)

    # 2. Physics & Engine Analytics
    res = compute_engine_analytics(pred_blend, rpm, load, fuel_flow, tau_ms)

    # 3. Print Report
    print("\n" + "=" * 75)
    print(f"       DIESEL-ISOBUTANOL PINN FULL DIAGNOSTIC REPORT: {title}")
    print("=" * 75)
    print(f" [ENGINE OPERATING POINT]")
    print(f"  • Speed: {rpm:.0f} RPM | Load: {load:.2f} | Fuel Flow: {fuel_flow:.3f} g/s | Tau: {tau_ms:.3f} ms")
    print(f"\n [PINN BLEND ESTIMATE]")
    print(f"  • Predicted Isobutanol: {res['blend_pct']:.2f}%")
    print(f"  • Effective LHV:        {res['effective_lhv_mj_kg']} MJ/kg  (Neat Diesel: 42.6 MJ/kg)")
    print(f"  • Blended Cetane:       {res['blended_cetane']}        (Neat Diesel: 51.0)")
    print(f"\n [FUEL ECONOMY / MILEAGE IMPACT]")
    print(f"  • Mileage Loss:         -{res['mileage_loss_pct']}%")
    print(f"  • BSFC Fuel Mass Rise:  +{res['bsfc_increase_pct']}%")
    print(f"\n [EMISSIONS PROFILE]")
    print(f"  • Soot / PM Reduction:  -{res['soot_reduction_pct']}% (Clean combustion benefit)")
    print(f"  • NOx Delta:            +{res['nox_shift_pct']}%")
    print(f"  • Direct Tailpipe CO2:  {res['co2_delta_pct']}%")
    print(f"\n [MECHANICAL HEALTH & STRESS]")
    print(f"  • Combustion Stress:    {res['stress_level']} (Est. dP/dθ: {res['dp_dtheta_mpa_deg']} MPa/deg)")
    print(f"    -> {res['stress_description']}")
    print(f"  • HPFP Lubricity Risk:  {res['wear_risk']} (HFRR Wear Scar: {res['hfrr_wear_scar_um']} µm)")
    print(f"    -> {res['wear_description']}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    # Test on a known high-blend row from dataset
    run_diagnostics(row_idx=2)
    # Test on neat diesel row
    run_diagnostics(row_idx=0)