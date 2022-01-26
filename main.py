#%%
from cv2 import VideoWriter
import jax
import jax.numpy as jnp
import PIL
from matplotlib.colors import to_rgba
import requests
import io
import matplotlib.pyplot as plt
import jax.scipy as jsp
from jax import lax
from jax import vmap
from jax import random
from tqdm import tqdm

from flax.core.frozen_dict import FrozenDict
from flax import linen as nn
from flax.training import train_state
import optax
import ml_collections

from utils import load_emoji
from utils import VideoWriter

#%%
EPOCHS = 2000
LR = 1e-3

#%%
TARGET_SIZE = 40
TARGET_EMOJI = "🦎"
target_img = load_emoji(TARGET_EMOJI)
print(target_img.shape)
plt.imshow(target_img)

# %%
pool = jnp.zeros((1,40,40,16))
pool = pool.at[:,20,20,3:].set(1.0)

#%%
class CA3(nn.Module):
  
  @nn.compact
  def __call__(self, img):
      x = nn.Conv(features=128, kernel_size=(3,3), kernel_init=nn.initializers.glorot_uniform(), bias_init=nn.initializers.zeros, padding='SAME')(img)
      x = nn.relu(x)
      x = nn.Conv(features=16, kernel_size=(1,1), kernel_init=nn.initializers.zeros, use_bias=False)(x) 
      x = img + x
      return x

key = jax.random.PRNGKey(0)
ca3 = CA3()
params_init = ca3.init(key, pool[:4,...])['params']
output = ca3.apply({'params': params_init}, pool[:4,...])
output.shape

#  %%
ca=CA3()
state = train_state.TrainState.create(
    apply_fn = ca.apply,
    params = params_init,
    tx = optax.adam(LR))

# %%
@jax.jit
def apply_model(state, img):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params, x):
        def call_ca(_, x): return ca.apply({'params': params}, x)
        x = lax.fori_loop(0, 50, call_ca, x)
        loss = jnp.mean(jnp.square(x[...,:4]-target_img))
        return loss, x

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, x), grads = grad_fn(state.params, img)
    grads = jax.tree_map(lambda x: x/(jnp.linalg.norm(x)+1e-8), grads)
    state = state.apply_gradients(grads=grads)
    return state, grads, loss, x

# %%
key = random.PRNGKey(0)
losses = []
for idx_epochs in tqdm(range(EPOCHS)):
    
    x = pool
    state, grads, loss, x = apply_model(state, x)
    losses.append(loss)

    if idx_epochs == EPOCHS//2:
        LR = LR * 0.1
        state = train_state.TrainState.create(
        apply_fn = ca.apply,
        params = state.params,
        tx = optax.adam(LR)) 

# %%
print(losses[-1])
plt.semilogy(losses, label=f'epochs = {EPOCHS}')
plt.legend()

# %%
x = pool
with VideoWriter(filename = 'synthesis.mp4') as vid:
    for i in range(100):
        vid.add(x[0,...,:3])
        x = ca.apply({'params': state.params}, x)

# %%
x.shape, to_rgba(x[0])
# %%
