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

DATA_PATH = "000138/sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
HISTORY_BINS = 10 # 200ms Context

def create_lagged_features(data, bins):
    n_samples, n_channels = data.shape
    padded = np.vstack([np.zeros((bins, n_channels)), data])
    features = []
    for i in range(bins):
        features.append(padded[i : i + n_samples])
    return np.hstack(features)

def run_system():
    print("=== ACTIVATING CLOSED-LOOP HOLO-NEURAL STACK (WITH HISTORY) ===")
    
    loader = NLBLoader(DATA_PATH)
    raw_spikes = loader.get_training_data()
    raw_vel = loader.get_velocity_data()
    
    # 1. PREPROCESS: Create History Buffer
    print(f"[Pre] Stacking {HISTORY_BINS} frames of history...")
    lagged_spikes = create_lagged_features(raw_spikes, HISTORY_BINS)
    
    # Split
    split = int(len(lagged_spikes) * 0.5)
    X_train, Y_train = lagged_spikes[:split], raw_vel[:split]
    X_test, Y_test = lagged_spikes[split:], raw_vel[split:]
    
    # Train
    print("[1] Calibrating Active Decoder (Day 0)...")
    key = jax.random.PRNGKey(0)
    model = ActiveDecoder.train(key, jnp.array(X_train), jnp.array(Y_train))
    
    # INFERENCE LOOP
    print(f"[2] Running Closed-Loop Inference on {len(X_test)} frames...")
    
    ctrl_state = model.controller 
    start_t = time.time()
    
    # JAX Scan for Speed
    def step_fn(carry, x):
        ctrl = carry
        vel, new_ctrl = model(x, ctrl)
        return new_ctrl, vel
        
    final_ctrl, pred_vels = jax.lax.scan(step_fn, ctrl_state, jnp.array(X_test))
    
    duration = time.time() - start_t
    print(f"    Inference Time: {duration:.2f}s ({len(X_test)/duration:.0f} Hz)")
    
    # Evaluate
    r2 = r2_score(Y_test, pred_vels)
    print(f"\n>>> FINAL CLOSED-LOOP SCORE: R^2 = {r2:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test[1000:1500, 0], 'k-', alpha=0.5, label='True Velocity')
    plt.plot(pred_vels[1000:1500, 0], 'g-', linewidth=2, label=f'ApexNeuro Active (R2={r2:.2f})')
    plt.title("ApexNeuro: History Buffer + Active Gain + ODE Lag Removal")
    plt.legend()
    plt.savefig("results/active_kill_shot_final.png")
    print("    Graph saved to results/active_kill_shot_final.png")

if __name__ == "__main__":
    run_system()