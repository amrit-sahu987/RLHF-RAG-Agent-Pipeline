import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

# Import the modules we defined earlier
from network import ActorCritic
from train import update_step

# --- 1. MOCKING THE RUST BACKEND ---
def get_dummy_batch(batch_size=128, state_dim=128, action_dim=10):
    """
    Simulates the data we will eventually pull from the Rust Replay Buffer.
    Standard NumPy is fine here; JAX will convert it when it hits the JIT function.
    """
    # Fake embedding vectors from the "database"
    states = np.random.randn(batch_size, state_dim).astype(np.float32)
    
    # Fake actions (document IDs chosen by the agent)
    actions = np.random.randint(0, action_dim, size=(batch_size,), dtype=np.int32)
    
    # Fake probabilities of the actions when they were taken
    log_probs_old = np.log(np.random.uniform(0.1, 1.0, size=(batch_size,))).astype(np.float32)
    
    # Fake rewards/returns (e.g., +1 for correct doc, -1 for wrong doc)
    returns = np.random.randn(batch_size).astype(np.float32)
    
    # Fake advantages (Returns minus the Baseline Value)
    advantages = np.random.randn(batch_size).astype(np.float32)
    
    return (states, actions, log_probs_old, returns, advantages)

# --- 2. INITIALIZATION ---
def main():
    print("Initializing JAX PPO Agent...")
    
    # Define our dimensions
    STATE_DIM = 128
    ACTION_DIM = 10
    
    # Initialize random number generator
    rng = jax.random.PRNGKey(42)
    
    # Initialize the network
    model = ActorCritic(action_dim=ACTION_DIM)
    dummy_state = jnp.zeros((1, STATE_DIM)) 
    params = model.init(rng, dummy_state)['params']
    
    # Initialize the optimizer (eps=1e-5 is a PPO-specific stability trick)
    optimizer = optax.adam(learning_rate=3e-4, eps=1e-5)
    opt_state = optimizer.init(params)
    
    # Pack the state
    train_state = {
        'params': params,
        'opt_state': opt_state
    }
    
    print("Initialization Complete. Starting Dummy Training Loop...\n")
    
    # --- 3. THE TRAINING LOOP ---
    NUM_UPDATES = 101
    
    # Trigger the JIT compiler before the loop so we don't measure compile time
    print("Compiling JIT function (this takes a few seconds)...")
    dummy_batch = get_dummy_batch(batch_size=128, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    _ = update_step(train_state, model.apply, optimizer, dummy_batch)
    print("Compilation finished!\n")
    
    start_time = time.time()
    
    for i in range(NUM_UPDATES):
        # 1. Pull fake data from our "Rust Buffer"
        batch = get_dummy_batch(batch_size=128, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        
        # 2. Push it through the JAX XLA compiler
        train_state, metrics = update_step(train_state, model.apply, optimizer, batch)
        
        # 3. Logging
        if i % 10 == 0:
            actor_loss, critic_loss, entropy = metrics
            print(f"Step {i:3d} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Entropy: {entropy:.4f}")

    end_time = time.time()
    print(f"\nTraining complete! 100 updates took {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()