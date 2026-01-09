import jax
import jax.numpy as jnp
import equinox as eqx
from apexneuro.decoder.spike_encoder import SpikeHDCEncoder
from apexneuro.core.control_loop import IntegralController

class ActiveDecoder(eqx.Module):
    encoder: SpikeHDCEncoder
    readout: jnp.ndarray
    controller: IntegralController
    
    def __init__(self, key, input_dim, hyper_dim=10000, target_energy=1.0):
        # 1. HDC Encoder (The Chassis)
        self.encoder = SpikeHDCEncoder.create(key, input_dim, hyper_dim)
        
        # 2. Linear Readout (The Steering)
        self.readout = jnp.zeros((hyper_dim, 2))
        
        # 3. Integral Controller (The Engine Governor)
        # Sets the target energy to the "Healthy" baseline
        self.controller = IntegralController(setpoint=target_energy)

    def __call__(self, spike_window, controller_state):
        # 1. Encode History Window
        hv_raw = self.encoder.encode_batch(spike_window[None, :]).squeeze()
        
        # 2. Measure Energy (Norm / sqrt(D))
        energy = jnp.linalg.norm(hv_raw) / jnp.sqrt(self.encoder.hyper_dim)
        
        # 3. FEEDBACK: Update Gain
        # If signal is weak (Day 40), Gain goes UP.
        new_ctrl, gain = controller_state.update(energy)
        
        # 4. Apply Correction
        hv_corrected = hv_raw * gain
        
        # 5. Readout (No Random ODE Distortion)
        velocity = hv_corrected @ self.readout
        
        return velocity, new_ctrl

    @classmethod
    def train(cls, key, X, Y, hyper_dim=10000):
        # 1. Measure Baseline Energy (Day 0)
        temp_encoder = SpikeHDCEncoder.create(key, X.shape[1], hyper_dim)
        H_raw = temp_encoder.encode_batch(X)
        
        energies = jnp.linalg.norm(H_raw, axis=1) / jnp.sqrt(hyper_dim)
        avg_energy = jnp.mean(energies)
        print(f"   [Calibration] Healthy Baseline Energy: {avg_energy:.4f}")
        
        # 2. Initialize Model
        model = cls(key, X.shape[1], hyper_dim, target_energy=avg_energy)
        
        # 3. Solve Weights (Ridge)
        lambda_reg = 100.0
        cov = H_raw.T @ H_raw + lambda_reg * jnp.eye(hyper_dim)
        proj = H_raw.T @ Y
        weights = jnp.linalg.solve(cov, proj)
        
        return eqx.tree_at(lambda m: m.readout, model, weights)