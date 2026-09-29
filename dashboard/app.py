import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model import DieselPINN
from dataset import FEATURE_COLS, TARGET_COL
from engine_analytics import compute_engine_analytics

# Page Configuration
st.set_page_config(
    page_title="Diesel-Isobutanol PINN Diagnostic Suite",
    page_icon="🚜",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        padding: 18px;
        background-color: #1e2130;
        border-radius: 8px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #adb5bd;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8f9fa;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .delta-neg { color: #ea868f; }
    .delta-pos { color: #75b798; }
    .delta-warn { color: #ffda6a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    weights_path = os.path.join("models", "diesel_pinn_weights.pt")
    scaler_path = os.path.join("models", "diesel_scaler.pkl")

    if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
        st.error("Model artifacts not found under `models/`. Run `python src/train.py` first.")
        st.stop()

    scaler = joblib.load(scaler_path)
    model = DieselPINN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

@st.cache_data
def load_dataset():
    data_path = os.path.join("data", "diesel_isobutanol_augmented_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model, scaler = load_model_artifacts()
df = load_dataset()

# --- HEADER ---
st.title("🚜 Diesel-Isobutanol PINN Diagnostic & Telemetry Suite")
st.caption("Physics-Informed Neural Network (PINN) for Compression-Ignition blend fraction determination and thermochemical stress diagnostics.")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Telemetry Source")
mode = st.sidebar.radio("Input Mode:", ["Dataset Record Lookup", "Manual Custom Input"])

actual_blend = None

if mode == "Dataset Record Lookup" and df is not None:
    st.sidebar.markdown(f"**Records Available:** `{len(df)}` rows")
    row_idx = st.sidebar.number_input("Select Dataset Row Index", min_value=0, max_value=len(df)-1, value=2, step=1)
    
    # Direct read from dataset
    row_data = df.iloc[row_idx]
    default_rpm = float(row_data['engine_rpm'])
    default_load = float(row_data['engine_load'])
    default_fuel = float(row_data['fuel_flow_gps'])
    default_tau = float(row_data['ignition_delay_est_ms'])
    actual_blend = float(row_data[TARGET_COL])
else:
    default_rpm = 2500.0
    default_load = 0.60
    default_fuel = 2.45
    default_tau = 1.65

# --- SECTION 1: 4 CORE TELEMETRY INPUTS ---
st.subheader("1. Ingested Engine Telemetry (4 Core Inputs)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    rpm = st.number_input("Engine Speed (RPM)", min_value=500.0, max_value=6500.0, value=default_rpm, step=50.0)
with col2:
    load = st.number_input("Engine Load (0.05 - 1.00)", min_value=0.05, max_value=1.00, value=default_load, step=0.05)
with col3:
    fuel_flow = st.number_input("Fuel Flow Rate (g/s)", min_value=0.05, max_value=15.0, value=default_fuel, step=0.05)
with col4:
    tau_ms = st.number_input("Ignition Delay τ (ms)", min_value=0.20, max_value=10.0, value=default_tau, step=0.05)

# --- PINN PREDICTION ---
input_array = np.array([[rpm, load, fuel_flow, tau_ms]])
scaled_input = scaler.transform(input_array)
with torch.no_grad():
    raw_pred = model(torch.tensor(scaled_input, dtype=torch.float32)).item()
predicted_blend = max(0.0, float(raw_pred))

st.markdown("---")

# --- SECTION 2: BLEND DETERMINATION (ACTUAL BESIDE PREDICTED) ---
st.subheader("2. Fuel Blend Determination")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if actual_blend is not None:
        st.metric("Actual Isobutanol Blend", f"{actual_blend:.2f}%")
    else:
        st.metric("Actual Isobutanol Blend", "N/A (Custom Test)")

with col_p2:
    st.metric("PINN Predicted Blend", f"{predicted_blend:.2f}%")

with col_p3:
    if actual_blend is not None:
        err = abs(actual_blend - predicted_blend)
        st.metric("Absolute Prediction Error", f"{err:.2f}%", delta=f"{'-' if err < 0.5 else '+'}{err:.2f}%", delta_color="inverse")
    else:
        classification = "Neat Diesel (B0)" if predicted_blend < 0.5 else f"Isobutanol Blend (~B{int(round(predicted_blend))})"
        st.metric("Fuel Classification", classification, delta="Physics Validated")

st.markdown("---")

# --- SECTION 3: ANALYZE BUTTON & RESULTS ---
st.subheader("3. Comprehensive Analytical Diagnostics")
run_analysis = st.button("🚀 Analyze Engine Diagnostics", type="primary", use_container_width=True)

if run_analysis:
    analytics = compute_engine_analytics(predicted_blend, rpm, load, fuel_flow, tau_ms)
    
    st.markdown("#### Performance, Emissions & Mechanical Stress Matrix")
    
    # Row 1: Mileage & Emissions
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Mileage / Range Delta</div>
            <div class="metric-value">-{analytics['mileage_loss_pct']:.2f}%</div>
            <div class="metric-delta delta-neg">LHV: {analytics['effective_lhv_mj_kg']} MJ/kg</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">BSFC Fuel Mass Demand</div>
            <div class="metric-value">+{analytics['bsfc_increase_pct']:.2f}%</div>
            <div class="metric-delta delta-warn">Fuel compensation factor</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Particulate (Soot) Reduction</div>
            <div class="metric-value">-{analytics['soot_reduction_pct']:.1f}%</div>
            <div class="metric-delta delta-pos">Fuel-bound oxygen benefit</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">NOx Emission Shift</div>
            <div class="metric-value">+{analytics['nox_shift_pct']:.1f}%</div>
            <div class="metric-delta delta-warn">Thermal premixed combustion</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Mechanical Stress & Reliability
    s1, s2, s3 = st.columns(3)
    stress_color = "delta-pos" if analytics['stress_level'] == "NORMAL" else ("delta-warn" if analytics['stress_level'] == "MODERATE" else "delta-neg")
    wear_color = "delta-pos" if analytics['wear_risk'] == "LOW" else ("delta-warn" if analytics['wear_risk'] == "MODERATE" else "delta-neg")

    with s1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Combustion Roughness (dP/dθ)</div>
            <div class="metric-value">{analytics['dp_dtheta_mpa_deg']:.3f} <span style="font-size:1rem;">MPa/deg</span></div>
            <div class="metric-delta {stress_color}">Stress Level: <b>{analytics['stress_level']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">HPFP & Injector Lubricity</div>
            <div class="metric-value">{analytics['hfrr_wear_scar_um']:.1f} <span style="font-size:1rem;">µm HFRR</span></div>
            <div class="metric-delta {wear_color}">Wear Risk: <b>{analytics['wear_risk']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        cetane_delta = analytics['blended_cetane'] - 51.0
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Blended Cetane Number</div>
            <div class="metric-value">{analytics['blended_cetane']:.1f}</div>
            <div class="metric-delta delta-neg">{cetane_delta:.1f} vs Neat Diesel (51.0)</div>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 4: DETAILED EXPLANATIONS ---
    st.markdown("---")
    st.subheader("4. Detailed Metric Explanations & Physical Insights")

    with st.expander("⛽ 1. Mileage Drop & BSFC Increase (Thermodynamics of Lower Heating Value)", expanded=True):
        st.markdown(f"""
        * **What it means:** Neat ultra-low sulfur diesel (ULSD) has a Lower Heating Value (LHV) of **42.6 MJ/kg** and density of **832 kg/m³**, delivering **~35.44 MJ of energy per liter**. Isobutanol has an LHV of **33.3 MJ/kg** and density of **802 kg/m³** (~26.71 MJ/L).
        * **The Result:** At your predicted blend of **{predicted_blend:.2f}% isobutanol**, the blended volumetric energy density decreases, resulting in an expected mileage penalty of **{analytics['mileage_loss_pct']:.2f}%**.
        * **Engine Controller (ECU) Response:** To produce equivalent brake torque and maintain vehicle speed, the ECU extends the injection pulse-width, causing Brake Specific Fuel Consumption (BSFC) to increase by **{analytics['bsfc_increase_pct']:.2f}%**.
        """)

    with st.expander("💨 2. Emissions Profile (Soot-NOx Trade-Off Dynamics)", expanded=True):
        st.markdown(f"""
        * **Soot / PM Reduction (-{analytics['soot_reduction_pct']:.1f}%):** Isobutanol ($C_4H_{{10}}O$) incorporates **~21.6% fuel-bound oxygen by weight**. In compression-ignition engines, soot typically forms in local, fuel-rich diffusion flame cores. The presence of oxygen inside the injected droplets promotes in-cylinder soot oxidation and suppresses polycyclic aromatic hydrocarbon (PAH) precursors, substantially cutting tailpipe smoke and particulate emissions.
        * **NOx Shift (+{analytics['nox_shift_pct']:.1f}%):** Isobutanol depresses the blend's cetane number from 51 down to **{analytics['blended_cetane']:.1f}**. A lower cetane number extends the ignition delay ($\\tau_{{id}}$), allowing more fuel to vaporize and premix with air before auto-ignition occurs. Once combustion initiates, the combustion of this larger premixed mass drives high local heat-release rates and peak flame temperatures, slightly increasing thermal $NO_x$ formation via the extended Zeldovich mechanism.
        """)

    with st.expander("⚙️ 3. Mechanical Stress & Combustion Roughness ($dP/d\\theta$)", expanded=True):
        st.markdown(f"""
        * **Combustion Noise & Peak Pressure Rise:** The estimated pressure rise rate is **{analytics['dp_dtheta_mpa_deg']:.3f} MPa/deg**. Standard diesel engines run smoothly between **0.40 and 0.60 MPa/deg**.
        * **Current Assessment:** **{analytics['stress_description']}**
        * **Mechanical Impact:** When ignition delay is stretched, rapid combustion of the premixed charge creates steep pressure gradients ($dP/d\\theta$). If this exceeds 0.85 MPa/deg, the resulting mechanical shock creates harsh acoustic diesel knock and exerts elevated peak loads on connecting rod small-end bushings, piston crowns, and wrist pins.
        """)

    with st.expander("🛡️ 4. Fuel System Lubricity & High-Pressure Fuel Pump (HPFP) Longevity", expanded=True):
        st.markdown(f"""
        * **Lubricity & Boundary Film Deficit:** High-pressure common rail fuel pumps (Bosch CP4, Denso HP3/4) rely on the fuel itself for internal lubrication under extreme operating pressures ($1,600 - 2,200\\text{{ bar}}$).
        * **HFRR Wear Scar Diameter:** ASTM D975 / EN 590 diesel standards require a High-Frequency Reciprocating Rig (HFRR) wear scar diameter below **460 µm**. Your blend shows an estimated wear scar of **{analytics['hfrr_wear_scar_um']:.1f} µm** (Wear Risk: **{analytics['wear_risk']}**).
        * **Recommendation:** **{analytics['wear_description']}** Adding commercial biodiesel (FAME) at 1–2% or a specialized lubricity additive completely mitigates this risk.
        """)