import jax
import optax
from functools import partial

# Import the loss function we defined in loss.py
from loss import ppo_loss

# We use partial to tell JIT to treat apply_fn and optimizer as static (non-tracing) variables
@partial(jax.jit, static_argnums=(1, 2))
def update_step(train_state, apply_fn, tx, batch):
    """
    train_state: A dictionary or Flax TrainState containing the network parameters
    apply_fn: The forward pass function of our ActorCritic model
    tx: The optax optimizer
    batch: The tuple of arrays from our replay buffer
    """
    
    # value_and_grad computes both the loss and the gradients of the loss w.r.t parameters
    loss_fn = partial(ppo_loss, apply_fn=apply_fn, batch=batch)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state['params'])
    
    # Calculate the optimizer updates and apply them to the parameters
    updates, new_opt_state = tx.update(grads, train_state['opt_state'], train_state['params'])
    new_params = optax.apply_updates(train_state['params'], updates)
    
    # Return the NEW, immutable state
    new_train_state = {
        'params': new_params,
        'opt_state': new_opt_state
    }
    
    return new_train_state, metrics