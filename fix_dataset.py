import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("data/diesel_isobutanol_augmented_dataset.csv")

# Constants
LHV_DIESEL = 42.6       # MJ/kg
LHV_ISOBUTANOL = 33.3   # MJ/kg
RHO_DIESEL = 832.0      # kg/m^3
RHO_ISOBUTANOL = 802.0  # kg/m^3
CETANE_DIESEL = 51.0
CETANE_ISOBUTANOL = 15.0

# 1. Realistic blend distribution across 0% to 20%
np.random.seed(42)
df['isobutanol_pct'] = np.random.choice([0.0, 5.0, 10.0, 15.0, 20.0], size=len(df), p=[0.3, 0.2, 0.2, 0.15, 0.15])
# Add slight continuous spread (+/- 0.5%)
df['isobutanol_pct'] = np.clip(df['isobutanol_pct'] + np.random.uniform(-0.5, 0.5, size=len(df)), 0.0, 22.0)
df['isobutanol_volume_fraction'] = df['isobutanol_pct'] / 100.0

# 2. Thermochemical properties
df['blend_density_kg_m3'] = (
    df['isobutanol_volume_fraction'] * RHO_ISOBUTANOL + 
    (1.0 - df['isobutanol_volume_fraction']) * RHO_DIESEL
)
df['isobutanol_mass_fraction'] = (
    df['isobutanol_volume_fraction'] * RHO_ISOBUTANOL
) / df['blend_density_kg_m3']

df['blend_lhv_mj_kg'] = (
    df['isobutanol_mass_fraction'] * LHV_ISOBUTANOL + 
    (1.0 - df['isobutanol_mass_fraction']) * LHV_DIESEL
)
df['blended_cetane_number'] = (
    df['isobutanol_volume_fraction'] * CETANE_ISOBUTANOL + 
    (1.0 - df['isobutanol_volume_fraction']) * CETANE_DIESEL
)

# 3. Deterministic Physical Telemetry Coupling
# Baseline neat diesel fuel flow based on displacement/load/rpm
base_fuel_flow = (df['engine_rpm'] / 1000.0) * df['engine_load'] * 1.2
energy_deficit_ratio = LHV_DIESEL / df['blend_lhv_mj_kg']
df['fuel_flow_gps'] = base_fuel_flow * energy_deficit_ratio

# Deterministic ignition delay (Arrhenius / Wolfer relation proxy)
df['ignition_delay_est_ms'] = (
    (55.0 / df['blended_cetane_number']) ** 0.9 * 
    (1.1 / (df['engine_load'] + 0.15))
)

# Derived metrics
df['bsfc_delta_pct'] = (energy_deficit_ratio - 1.0) * 100.0
df['soot_pm_reduction_factor'] = np.clip(1.0 - (df['isobutanol_pct'] * 0.035), 0.35, 1.0)

df.to_csv("data/diesel_isobutanol_augmented_dataset.csv", index=False)
print("Dataset re-generated with clean physical determinism.")