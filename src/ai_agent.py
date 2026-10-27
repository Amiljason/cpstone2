import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def get_agent_response(chat_history, user_message, live_telemetry_context):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "⚠️ Gemini API key is missing."

    client = genai.Client(api_key=key)

    system_instruction = f"""
    You are an expert Automotive Diagnostic AI and Powertrain Calibration Engineer assisting with a 
    Diesel-Isobutanol Physics-Informed Neural Network (PINN) monitoring system.

    CURRENT VEHICLE LIVE CONTEXT:
    - Engine RPM: {live_telemetry_context.get('rpm')} RPM
    - Engine Load: {live_telemetry_context.get('load')}
    - Fuel Flow: {live_telemetry_context.get('fuel_flow')} g/s
    - Ignition Delay (tau): {live_telemetry_context.get('tau_ms')} ms
    - PINN Predicted Isobutanol Blend: {live_telemetry_context.get('predicted_blend'):.2f}%
    - Mileage Delta: -{live_telemetry_context.get('mileage_loss'):.2f}%
    - Soot Reduction: -{live_telemetry_context.get('soot_reduction'):.1f}%
    - NOx Delta: +{live_telemetry_context.get('nox_shift'):.1f}%
    - Combustion Roughness (dP/dtheta): {live_telemetry_context.get('dp_dtheta')} MPa/deg ({live_telemetry_context.get('stress_level')})
    - HPFP Wear Scar: {live_telemetry_context.get('hfrr_wear_scar')} um ({live_telemetry_context.get('wear_risk')} Risk)
    - Blended Cetane: {live_telemetry_context.get('cetane')}

    YOUR ROLE:
    1. Answer driver and engineer questions accurately based on compression-ignition thermodynamics and real-time telemetry.
    2. Explain physical phenomena simply (cetane depression, lubricity depletion, soot oxidation, pressure rise rates).
    3. Give practical advice regarding additive dosing, fuel filters, cold starting, and driving style.
    4. Keep answers concise, direct, and free of fluff.
    """

    contents = []
    for msg in chat_history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    configured_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    models_to_try = [configured_model, "gemini-3.5-flash"]
    
    errors = []

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return response.text
        except Exception as e:
            errors.append(f"{m}: {e}")
            continue

    return "Error contacting AI Agent: " + " | ".join(errors)