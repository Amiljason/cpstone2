import os
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import DieselDataset, FEATURE_COLS
from model import DieselPINN
from physics_loss import DieselIsobutanolPINNLoss

def run_training():
    data_path = os.path.join("data", "diesel_isobutanol_augmented_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at {data_path}. Place it there first.")

    # 1. Load Data & Split
    print("Loading augmented dataset...")
    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. Datasets & DataLoaders
    train_ds = DieselDataset(train_df, is_train=True)
    val_ds = DieselDataset(val_df, scaler=train_ds.scaler, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 3. Model, Loss & Optimizer Initialization
    model = DieselPINN(input_dim=len(FEATURE_COLS))
    criterion = DieselIsobutanolPINNLoss(lambda_energy=0.01, lambda_ignition=0.05)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 80
    print(f"Beginning Diesel PINN Training ({epochs} epochs)...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch['x'])

            raw_telemetry = {
                'engine_load': batch['engine_load'].unsqueeze(1),
                'fuel_flow_gps': batch['fuel_flow_gps'].unsqueeze(1),
                'ignition_delay_est_ms': batch['ignition_delay_est_ms'].unsqueeze(1)
            }

            loss, l_sup, l_eng, l_ign = criterion(preds, batch['y'], raw_telemetry)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch['x'])
                raw_telemetry = {
                    'engine_load': batch['engine_load'].unsqueeze(1),
                    'fuel_flow_gps': batch['fuel_flow_gps'].unsqueeze(1),
                    'ignition_delay_est_ms': batch['ignition_delay_est_ms'].unsqueeze(1)
                }
                v_loss, _, _, _ = criterion(preds, batch['y'], raw_telemetry)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 4. Save Artifacts
    os.makedirs("models", exist_ok=True)
    weights_path = os.path.join("models", "diesel_pinn_weights.pt")
    scaler_path = os.path.join("models", "diesel_scaler.pkl")

    torch.save(model.state_dict(), weights_path)
    joblib.dump(train_ds.scaler, scaler_path)
    print(f"\nArtifacts saved successfully:")
    print(f" -> {weights_path}")
    print(f" -> {scaler_path}")

if __name__ == "__main__":
    run_training()