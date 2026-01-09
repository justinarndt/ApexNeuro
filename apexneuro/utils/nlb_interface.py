"""
APEXNEURO - NLB INTERFACE (MULTI-INDEX FIX)
Bridges the 'Neural Latents Benchmark' NWB format to JAX arrays.
Fixes: Pandas 2.0+ compatibility, Argument Names, and MultiIndex Column Selection.
"""

import numpy as np
import jax.numpy as jnp
from nlb_tools.nwb_interface import NWBDataset
import os
import pandas as pd

class NLBLoader:
    def __init__(self, datapath: str, bin_width_ms: int = 20):
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Data not found at: {datapath}")
            
        print(f"   [Loader] Parsing NWB file: {os.path.basename(datapath)}...")
        self.dataset = NWBDataset(datapath)
        self.bin_width = bin_width_ms
        
        # --- PATCH: MANUAL RESAMPLING (Pandas 2.0+ Fix) ---
        print(f"   [Loader] Resampling to {bin_width_ms}ms (Manual Patch)...")
        df = self.dataset.data
        bin_str = f"{bin_width_ms}ms"
        
        # ROBUST COLUMN SELECTION (Handles MultiIndex Tuples)
        # We look for 'spikes' in the first level of the column tuple (e.g., 'heldout_spikes')
        spike_cols = []
        other_cols = []
        
        for c in df.columns:
            # Check if column is a tuple (MultiIndex) or string
            col_name = c[0] if isinstance(c, tuple) else c
            
            if 'spikes' in str(col_name):
                spike_cols.append(c)
            else:
                other_cols.append(c)
        
        # Resample separately
        resampled_spikes = df[spike_cols].resample(bin_str).sum()
        resampled_others = df[other_cols].resample(bin_str).mean()
        
        # Recombine
        new_df = pd.concat([resampled_spikes, resampled_others], axis=1)
        
        self.dataset.data = new_df
        self.dataset.bin_width = bin_width_ms
        # --------------------------------

    def get_training_data(self):
        """
        Extracts the full spike matrix (Time x Channels).
        """
        # Generate trial-aligned data
        # Note: We use 'move_onset_time' which we confirmed exists
        data_df = self.dataset.make_trial_data(align_field='move_onset_time', align_range=(-100, 400))
        
        # Select spike columns again from the trial dataframe
        spike_cols = []
        for c in data_df.columns:
            col_name = c[0] if isinstance(c, tuple) else c
            if 'spikes' in str(col_name):
                spike_cols.append(c)
                
        spike_matrix = data_df[spike_cols].values
        
        print(f"   [Loader] Extracted Spikes: {spike_matrix.shape}")
        if spike_matrix.shape[1] == 0:
            raise ValueError("No spike columns found! Check column naming.")
            
        return jnp.array(spike_matrix, dtype=jnp.float32)

    def get_velocity_data(self):
        """
        Extracts hand velocity (Targets).
        """
        data_df = self.dataset.make_trial_data(align_field='move_onset_time', align_range=(-100, 400))
        
        vel_cols = []
        for c in data_df.columns:
            col_name = c[0] if isinstance(c, tuple) else c
            # Look for 'vel' (hand_vel, finger_vel, etc)
            if 'vel' in str(col_name):
                vel_cols.append(c)
        
        vel_matrix = data_df[vel_cols].values
        print(f"   [Loader] Extracted Velocity: {vel_matrix.shape}")
        return jnp.array(vel_matrix, dtype=jnp.float32)