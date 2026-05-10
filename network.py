import flax.linen as nn
import jax.numpy as jnp

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared MLP Base
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Critic Head (Outputs a single scalar value for the state)
        value = nn.Dense(1)(x)

        # Actor Head (Outputs logits for the action distribution)
        # Using a Categorical distribution for discrete actions (e.g., retrieving a document ID)
        logits = nn.Dense(self.action_dim)(x)

        return logits, jnp.squeeze(value, axis=-1)