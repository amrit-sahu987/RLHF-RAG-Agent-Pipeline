import jax
import jax.numpy as jnp
import optax

def ppo_loss(params, apply_fn, batch, clip_eps=0.2, val_coef=0.5, ent_coef=0.01):
    # Unpack the batch data retrieved from your Phase 2 Rust buffer
    states, actions, log_probs_old, returns, advantages = batch
    
    # Run the forward pass with the CURRENT parameters
    logits, values = apply_fn({'params': params}, states)
    
    # Calculate the new probabilities of the actions taken
    action_probs = jax.nn.softmax(logits)
    log_probs_new = jnp.log(jnp.take_along_axis(action_probs, actions[:, None], axis=1).squeeze())
    
    # 1. Actor Loss (Clipped Surrogate Objective)
    ratio = jnp.exp(log_probs_new - log_probs_old)
    p1 = ratio * advantages
    p2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -jnp.mean(jnp.minimum(p1, p2))
    
    # 2. Critic Loss (Value function error)
    critic_loss = jnp.mean((returns - values) ** 2)
    
    # 3. Entropy Bonus (Encourages exploration)
    entropy = -jnp.mean(jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1))
    
    # Total Loss
    total_loss = actor_loss + (val_coef * critic_loss) - (ent_coef * entropy)
    
    return total_loss, (actor_loss, critic_loss, entropy)