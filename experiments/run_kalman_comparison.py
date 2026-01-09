"""
RUN_KALMAN_COMPARISON.py (JAX ACCELERATED)
Head-to-Head: ApexNeuro vs. Adaptive Kalman Filter
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apexneuro.utils.nlb_interface import NLBLoader
from apexneuro.decoder.active_decoder import ActiveDecoder

# CONFIG
DATA_PATH = "000138/sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
HISTORY_BINS = 10 

# --- JAX-ACCELERATED KALMAN FILTER ---
class JaxKalmanFilter:
    def __init__(self, state_dim=2, obs_dim=162, adaptation_rate=0.001):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.alpha = adaptation_rate
        
        # Initialize Matrices (Identity for now)
        self.A = jnp.eye(state_dim)
        self.C = jnp.zeros((obs_dim, state_dim))
        self.Q = jnp.eye(state_dim) * 0.01
        self.R = jnp.eye(obs_dim) * 1.0
        
        # Initial State
        self.mu_x = jnp.zeros(state_dim)
        self.P = jnp.eye(state_dim)
        self.obs_mean = jnp.zeros(obs_dim)

    def fit(self, X, Y):
        print("   [Kalman] Fitting Observation Model (Day 0)...")
        # Standard OLS fit for C matrix
        # X=Observations (Spikes), Y=State (Velocity) -- Reversed for KF convention
        # We usually model Spikes = C * Velocity + Noise
        
        # Centering
        self.obs_mean = jnp.mean(X, axis=0)
        X_centered = X - self.obs_mean
        
        # Solve C: X = Y @ C.T  =>  C = (Y.T Y)^-1 Y.T X
        # Ridge term for stability
        lambda_reg = 100.0
        Y_cov = Y.T @ Y + lambda_reg * jnp.eye(self.state_dim)
        cross_cov = Y.T @ X_centered
        self.C = jnp.linalg.solve(Y_cov, cross_cov).T
        
        # Convert everything to JAX arrays
        self.A = jnp.array(self.A)
        self.Q = jnp.array(self.Q)
        self.R = jnp.array(self.R)
        self.mu_x = jnp.array(self.mu_x)
        self.P = jnp.array(self.P)
        self.obs_mean = jnp.array(self.obs_mean)

    def predict_batch(self, X_stream):
        # The Step Function (Compiled)
        def step_fn(carry, z_raw):
            mu_x, P, obs_mean = carry
            
            # 1. ADAPTATION (Unsupervised Bias Update)
            obs_mean_new = (1 - self.alpha) * obs_mean + self.alpha * z_raw
            z_centered = z_raw - obs_mean_new
            
            # 2. PREDICT (Time Update)
            mu_pred = self.A @ mu_x
            P_pred = self.A @ P @ self.A.T + self.Q
            
            # 3. UPDATE (Measurement Update)
            # Innovation Covariance S = C P C^T + R
            S = self.C @ P_pred @ self.C.T + self.R
            
            # Kalman Gain K = P C^T S^-1
            # We use solve(S, C P) instead of inversion for stability
            K_trans = jnp.linalg.solve(S, self.C @ P_pred).T
            
            # State Update
            y_pred = self.C @ mu_pred
            residual = z_centered - y_pred
            mu_new = mu_pred + K_trans @ residual
            
            # Covariance Update: P = (I - K C) P
            I = jnp.eye(self.state_dim)
            P_new = (I - K_trans @ self.C) @ P_pred
            
            return (mu_new, P_new, obs_mean_new), mu_new

        # Run Scan
        init_carry = (self.mu_x, self.P, self.obs_mean)
        _, preds = jax.lax.scan(step_fn, init_carry, X_stream)
        return preds

def create_history(data, bins):
    n, c = data.shape
    padded = np.vstack([np.zeros((bins, c)), data])
    return np.hstack([padded[i:i+n] for i in range(bins)])

def run_head_to_head():
    print("=== HEAD-TO-HEAD: APEXNEURO vs KALMAN (JAX) ===")
    
    loader = NLBLoader(DATA_PATH)
    raw_spikes = loader.get_training_data()
    raw_vel = loader.get_velocity_data()
    
    # Preprocessing
    lagged_spikes = create_history(raw_spikes, HISTORY_BINS)
    # Convert to JAX immediately
    lagged_spikes = jnp.array(lagged_spikes)
    raw_vel = jnp.array(raw_vel)
    
    split = int(len(lagged_spikes) * 0.5)
    
    X_train, Y_train = lagged_spikes[:split], raw_vel[:split]
    X_test, Y_test = lagged_spikes[split:], raw_vel[split:]
    
    # --- MODEL 1: ADAPTIVE KALMAN ---
    print("\n[1] Training Adaptive Kalman (Day 0)...")
    kf = JaxKalmanFilter(obs_dim=X_train.shape[1])
    kf.fit(X_train, Y_train)
    
    print(f"[2] Running Kalman Inference on Day 40 Drift ({len(X_test)} frames)...")
    start_k = time.time()
    # JIT Compile the predict function
    predict_fast = jax.jit(kf.predict_batch)
    kf_preds = predict_fast(X_test).block_until_ready()
    kf_time = time.time() - start_k
    r2_kf = r2_score(Y_test, kf_preds)
    print(f"    >> KALMAN R^2: {r2_kf:.4f} ({len(X_test)/kf_time:.0f} Hz)")
    
    # --- MODEL 2: APEXNEURO ---
    print("\n[3] Training ApexNeuro (Day 0)...")
    key = jax.random.PRNGKey(0)
    apex = ActiveDecoder.train(key, X_train, Y_train)
    
    print(f"[4] Running ApexNeuro Inference on Day 40 Drift...")
    start_a = time.time()
    
    @jax.jit
    def run_apex(x_data, ctrl_init):
        def step_fn(carry, x):
            ctrl = carry
            vel, new_ctrl = apex(x, ctrl)
            return new_ctrl, vel
        _, out = jax.lax.scan(step_fn, ctrl_init, x_data)
        return out
        
    apex_preds = run_apex(X_test, apex.controller).block_until_ready()
    apex_time = time.time() - start_a
    r2_apex = r2_score(Y_test, apex_preds)
    print(f"    >> APEX R^2:   {r2_apex:.4f} ({len(X_test)/apex_time:.0f} Hz)")
    
    # --- PLOT COMPARISON ---
    plt.figure(figsize=(14, 7))
    start, end = 1000, 1500
    
    plt.plot(Y_test[start:end, 0], 'k-', alpha=0.4, linewidth=3, label='True Velocity')
    plt.plot(kf_preds[start:end, 0], 'b--', linewidth=2, label=f'Adaptive Kalman (R2={r2_kf:.2f})')
    plt.plot(apex_preds[start:end, 0], 'g-', linewidth=2, label=f'ApexNeuro (R2={r2_apex:.2f})')
    
    plt.title("Head-to-Head: Active Gain vs. Adaptive Kalman")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/Head_to_Head.png")
    print("\nSaved Comparison Graph: assets/Head_to_Head.png")

if __name__ == "__main__":
    run_head_to_head()