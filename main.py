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

from utils import load_emoji, get_living_mask
from utils import VideoWriter

#%%
EPOCHS = 2000
LR_INIT = 1e-3
BATCH_SIZE=4 #for fast synthesis (but no persistance): BATCH_SIZE=1;x0=seed

#%%
TARGET_SIZE = 40
TARGET_EMOJI = "🦎"
target_img = load_emoji(TARGET_EMOJI)
target_img = target_img[None,:]
print(target_img.shape)
plt.imshow(target_img[0])

# %%
pool = jnp.zeros((256,40,40,16))
pool = pool.at[:,20,20,3:].set(1.0)
seed = jnp.zeros((1,40,40,16))
seed = seed.at[:,20,20,3:].set(1.0)

#%%
class CA3(nn.Module):
  
  @nn.compact
  def __call__(self, img):
      alive_mask = get_living_mask(img)
      x = nn.Conv(features=128, kernel_size=(3,3), kernel_init=nn.initializers.glorot_uniform(), bias_init=nn.initializers.zeros, padding='SAME')(img)
      x = nn.relu(x)
      x = nn.Conv(features=16, kernel_size=(1,1), kernel_init=nn.initializers.zeros, use_bias=False)(x) 
      x = img + x
      x *= alive_mask
      return x

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

#  %%
key = jax.random.PRNGKey(0)
ca=CA3()
params_init = ca.init(key, pool[:1,...])['params']
state = train_state.TrainState.create(
    apply_fn = ca.apply,
    params = params_init,
    tx = optax.adam(LR_INIT))

# %%
EPOCHS = 8000
LR = LR_INIT
key = random.PRNGKey(0)
losses = []
sks0 = random.split(key, EPOCHS)
for idx_epochs, sk0 in tqdm(enumerate(sks0), total=len(sks0)):
    
    idx_batch = random.choice(sk0, jnp.arange(0, len(pool), 1, dtype=int), (1,BATCH_SIZE), replace=False)[0]
    x0 = pool[idx_batch]
    x0 = x0.at[:1].set(seed)
    x0=seed
    state, grads, loss, x = apply_model(state, x0)
    losses.append(loss)

    pool = pool.at[idx_batch].set(x)

    if idx_epochs == EPOCHS//2:
        LR = LR * 0.1
        state = train_state.TrainState.create(
        apply_fn = ca.apply,
        params = state.params,
        tx = optax.adam(LR)) 

# %%
print(losses[-1])
plt.semilogy(losses, label=f'epochs = {EPOCHS}')
plt.ylim([1e-5,0])
plt.legend()

#%%
def to_rgb(x):
    rgb, alpha = x[...,:3], jnp.clip(x[...,3:4],0,1)
    return 1-alpha+rgb

# %%
x = seed
imgs_syn = []
with VideoWriter(filename = 'synthesis.mp4') as vid:
    for i in tqdm(range(100)):
        # im = jnp.clip(x[0,...,:4],0,1)
        im = jnp.clip(to_rgb(x)[0],0,1)
        vid.add(im)
        imgs_syn.append(im)
        x = ca.apply({'params': state.params}, x)

# %%
fig, ax = plt.subplots(4,12, figsize=(24,8))
for i in range(48):
    ax.flat[i].imshow(imgs_syn[i])
    # ax.flat[i].hist(imgs_syn[i].flatten())
    ax.flat[i].axis('off')
fig.tight_layout()
plt.savefig('figures/image_synthesis.png')
# %%