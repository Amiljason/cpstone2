def generate_recommendations(analytics):
    """
    Generates actionable maintenance, operating, and fueling advice
    based on predicted blend percentage and diagnostic stress levels.
    """
    blend = analytics['blend_pct']
    cetane = analytics['blended_cetane']
    stress = analytics['stress_level']
    wear_risk = analytics['wear_risk']
    dp_dtheta = analytics['dp_dtheta_mpa_deg']
    
    recs = []

    # 1. Fuel Lubricity & Additive Strategy
    if blend > 12.0:
        recs.append({
            "category": "Fuel System & Lubricity",
            "priority": "CRITICAL",
            "title": "Dose Biodiesel (B2) or High-Performance Lubricity Improver",
            "action": (
                f"Your blend ({blend:.1f}%) results in an HFRR wear scar of {analytics['hfrr_wear_scar_um']} µm, "
                "approaching the 460 µm critical pump limit. Dose 1–2% fatty acid methyl ester (biodiesel) or "
                "an ester-based lubricity additive (e.g., Stanadyne, Liqui Moly Diesel Lubricant) to prevent high-pressure "
                "fuel pump (HPFP) plunger scuffing."
            )
        })
    elif blend > 5.0:
        recs.append({
            "category": "Fuel System & Lubricity",
            "priority": "RECOMMENDED",
            "title": "Preventive Lubricity Treatment",
            "action": (
                "Mild boundary lubrication loss detected. Periodic dosing of commercial lubricity additive "
                "every 3–4 tank fills is suggested."
            )
        })
    else:
        recs.append({
            "category": "Fuel System & Lubricity",
            "priority": "NORMAL",
            "title": "Standard Diesel Lubricity",
            "action": "Fuel lubricity conforms to ASTM D975 / EN 590 standards. No additional additives needed."
        })

    # 2. Cold Start & Cetane Enhancers
    if cetane < 46.0:
        recs.append({
            "category": "Ignition & Cold Starts",
            "priority": "HIGH",
            "title": "Use Cetane Improver (2-EHN) or Extend Glow-Plug Pre-Heat",
            "action": (
                f"Blended cetane rating has dropped to {cetane:.1f} (vs neat diesel 51.0). In temperatures below 15°C, "
                "extend key-on glow plug duration before cranking. Consider adding 2-Ethylhexyl Nitrate (2-EHN) "
                "to raise cetane index and eliminate rough morning idling."
            )
        })
    else:
        recs.append({
            "category": "Ignition & Cold Starts",
            "priority": "NORMAL",
            "title": "Standard Auto-Ignition Timing",
            "action": "Cetane index is sufficient for prompt auto-ignition across normal operating temperature envelopes."
        })

    # 3. Mechanical Stress & Engine Operation
    if stress in ["HIGH / SEVERE", "MODERATE"]:
        recs.append({
            "category": "Engine Stress & Driving Style",
            "priority": "HIGH" if stress == "HIGH / SEVERE" else "MODERATE",
            "title": "Avoid High-Torque / Low-RPM Lugging",
            "action": (
                f"Combustion roughness index is {dp_dtheta:.3f} MPa/deg ({stress}). Downshift under incline "
                "or load rather than lugging below 1,600 RPM to reduce pressure spike shock loading on connecting rod "
                "wrist pins and cylinder head gaskets."
            )
        })
    else:
        recs.append({
            "category": "Engine Stress & Driving Style",
            "priority": "NORMAL",
            "title": "Combustion Envelopes Optimal",
            "action": "Pressure gradient rise rate is within standard smooth-burn thresholds (< 0.65 MPa/deg)."
        })

    # 4. Maintenance & Filter Intervals
    if blend > 10.0:
        recs.append({
            "category": "Maintenance Intervals",
            "priority": "MODERATE",
            "title": "Shorten Fuel Filter & Oil Change Intervals by 20–25%",
            "action": (
                "Isobutanol has higher hygroscopic potential and solvent properties that can loosen tank sediment and "
                "increase moisture accumulation. Inspect the fuel water separator every 5,000 km and replace the fuel filter "
                "earlier than the manufacturer schedule."
            )
        })

    return recs