import numpy as np

# Physical & Thermochemical Constants
LHV_DIESEL = 42.6          # MJ/kg
LHV_ISOBUTANOL = 33.3      # MJ/kg
RHO_DIESEL = 832.0         # kg/m^3
RHO_ISOBUTANOL = 802.0     # kg/m^3
CETANE_DIESEL = 51.0
CETANE_ISOBUTANOL = 15.0

def compute_engine_analytics(predicted_blend_pct, engine_rpm, engine_load, fuel_flow_gps, ignition_delay_est_ms):
    """
    Computes thermodynamic, emissions, and mechanical stress metrics
    given the predicted isobutanol blend percentage and engine operating conditions.
    """
    # Clamp blend percentage to realistic operational bounds
    v_b = float(np.clip(predicted_blend_pct / 100.0, 0.0, 0.40))
    blend_pct = v_b * 100.0

    # 1. Thermochemical Properties
    rho_blend = v_b * RHO_ISOBUTANOL + (1.0 - v_b) * RHO_DIESEL
    w_b = (v_b * RHO_ISOBUTANOL) / rho_blend
    effective_lhv = w_b * LHV_ISOBUTANOL + (1.0 - w_b) * LHV_DIESEL
    blended_cetane = v_b * CETANE_ISOBUTANOL + (1.0 - v_b) * CETANE_DIESEL

    # 2. Mileage & Fuel Economy Delta
    # Volumetric energy density comparison vs pure diesel
    neat_volumetric_energy = LHV_DIESEL * RHO_DIESEL
    blend_volumetric_energy = effective_lhv * rho_blend
    mileage_loss_pct = (1.0 - (blend_volumetric_energy / neat_volumetric_energy)) * 100.0
    bsfc_increase_pct = ((LHV_DIESEL / effective_lhv) - 1.0) * 100.0

    # 3. Emissions Shifts
    # Fuel-bound oxygen (~21.6 wt% in isobutanol) cuts soot precursor formation
    soot_reduction_pct = float(np.clip(blend_pct * 2.8, 0.0, 60.0))
    
    # NOx formation: prolonged ignition delay increases premixed burn fraction and peak temperatures
    nox_shift_pct = float(np.clip((blend_pct * 0.35) * (engine_load / 0.5), -2.0, 15.0))
    
    # CO2 tailpipe delta (lifecycle bio-credit not included, direct exhaust)
    co2_delta_pct = -float(blend_pct * 0.12)

    # 4. Mechanical Engine Stress: Combustion Roughness Index (dP/dθ proxy)
    # Baseline neat diesel ignition delay at this load
    tau_neat = (55.0 / CETANE_DIESEL) ** 0.9 * (1.1 / (engine_load + 0.15))
    tau_ratio = ignition_delay_est_ms / max(tau_neat, 1e-3)
    
    # dP/dtheta pressure rise rate proxy (MPa/deg crank angle)
    # Normal diesel runs ~0.4 to 0.6 MPa/deg; rough combustion is > 0.8 MPa/deg
    dp_dtheta_est = 0.45 * (tau_ratio ** 1.3) * (0.8 + 0.5 * engine_load)
    
    if dp_dtheta_est < 0.65:
        stress_level = "NORMAL"
        stress_desc = "Smooth combustion envelope, acceptable pressure rise rate."
    elif dp_dtheta_est < 0.85:
        stress_level = "MODERATE"
        stress_desc = "Noticeable diesel clatter; elevated premixed peak heat release."
    else:
        stress_level = "HIGH / SEVERE"
        stress_desc = "Excessive dP/dθ shock; high mechanical stress on wrist pins and piston crowns."

    # 5. Fuel System Lubricity & Component Wear Risk (HFRR wear scar proxy)
    # Isobutanol lacks boundary lubricating lubricity; neat diesel standard is < 460 µm
    hfrr_wear_scar_um = 310.0 + (blend_pct * 9.5)
    
    if blend_pct <= 5.0:
        wear_risk = "LOW"
        wear_desc = "Safe within factory HPFP and injector clearances."
    elif blend_pct <= 12.0:
        wear_risk = "MODERATE"
        wear_desc = "Lubricity additive recommended to prevent injector scuffing."
    else:
        wear_risk = "HIGH"
        wear_desc = "Severe HPFP lubricity deficit; risk of plunger seizure without lubricity dosing."

    return {
        "blend_pct": blend_pct,
        "effective_lhv_mj_kg": round(effective_lhv, 2),
        "blended_cetane": round(blended_cetane, 1),
        "mileage_loss_pct": round(mileage_loss_pct, 2),
        "bsfc_increase_pct": round(bsfc_increase_pct, 2),
        "soot_reduction_pct": round(soot_reduction_pct, 1),
        "nox_shift_pct": round(nox_shift_pct, 1),
        "co2_delta_pct": round(co2_delta_pct, 1),
        "dp_dtheta_mpa_deg": round(dp_dtheta_est, 3),
        "stress_level": stress_level,
        "stress_description": stress_desc,
        "hfrr_wear_scar_um": round(hfrr_wear_scar_um, 1),
        "wear_risk": wear_risk,
        "wear_description": wear_desc
    }