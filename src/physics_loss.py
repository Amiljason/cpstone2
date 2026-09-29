import torch
import torch.nn as nn

class DieselIsobutanolPINNLoss(nn.Module):
    def __init__(self, lambda_energy=0.05, lambda_ignition=0.08):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_energy = lambda_energy
        self.lambda_ignition = lambda_ignition
        
        self.lhv_diesel = 42.6       # MJ/kg
        self.lhv_isobutanol = 33.3   # MJ/kg
        self.cetane_diesel = 51.0
        self.cetane_isobutanol = 15.0
        self.rho_isobutanol = 802.0  # kg/m^3
        self.rho_diesel = 832.0      # kg/m^3

    def forward(self, pred_blend_pct, true_blend_pct, raw_telemetry):
        # 1. Supervised prediction loss
        l_supervised = self.mse(pred_blend_pct, true_blend_pct)
        
        # 2. Physics Prior: Cetane Depression & Ignition Delay Extension
        v_b = torch.clamp(pred_blend_pct / 100.0, 0.0, 0.30)
        blended_cetane = v_b * self.cetane_isobutanol + (1.0 - v_b) * self.cetane_diesel
        
        load = raw_telemetry['engine_load']
        expected_tau = (55.0 / (blended_cetane + 1e-5)) ** 0.8 * (1.2 / (load + 0.2))
        l_ignition = self.mse(expected_tau, raw_telemetry['ignition_delay_est_ms'])
        
        # 3. Physics Prior: Energy balance scaled to engine load
        w_b = (v_b * self.rho_isobutanol) / (
            v_b * self.rho_isobutanol + (1.0 - v_b) * self.rho_diesel + 1e-6
        )
        pred_lhv = w_b * self.lhv_isobutanol + (1.0 - w_b) * self.lhv_diesel
        
        expected_fuel_factor = self.lhv_diesel / pred_lhv
        specific_fuel = raw_telemetry['fuel_flow_gps'] / (load + 0.15)
        # Normalize specific fuel penalty
        l_energy = self.mse(expected_fuel_factor, 1.0 + (specific_fuel - specific_fuel.mean()) * 0.05)

        total_loss = l_supervised + (self.lambda_energy * l_energy) + (self.lambda_ignition * l_ignition)
        return total_loss, l_supervised, l_energy, l_ignition