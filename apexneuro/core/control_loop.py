import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class IntegralController:
    """
    Stabilizes Hypervector Energy to match the 'Day 0' baseline.
    """
    setpoint: float
    state: float = 0.0
    Ki: float = 0.01  # Lower gain for stability
    
    # SAFETY CLAMPS (Prevent Explosion)
    min_gain: float = 0.5
    max_gain: float = 5.0

    def update(self, current_energy: float):
        # Error: How far are we from the 'Healthy' baseline?
        error = self.setpoint - current_energy
        
        # Integrate (Accumulate correction)
        new_state = self.state + (error * self.Ki)
        
        # Convert to Gain (Exp scale for smooth multiplication)
        # Result is clamped between 0.5x and 5.0x
        raw_gain = jnp.exp(new_state)
        gain = jnp.clip(raw_gain, self.min_gain, self.max_gain)
        
        return self.replace(state=new_state), gain