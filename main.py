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

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils import load_emoji, get_living_mask
from utils import VideoWriter
from utils_figures import fig_loss_and_synthesis
from utils_percep import perception

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    #HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()

    target_img = load_emoji(cfg.TARGET_EMOJI)
    target_img = target_img[None,:]

    pool = jnp.zeros((256,cfg.TARGET_SIZE,cfg.TARGET_SIZE,cfg.TARGET_CH))
    pool = pool.at[:,cfg.TARGET_SIZE//2,cfg.TARGET_SIZE//2,3:].set(1.0)
    seed = jnp.zeros((1,cfg.TARGET_SIZE,cfg.TARGET_SIZE,cfg.TARGET_CH))
    seed = seed.at[:,cfg.TARGET_SIZE//2,cfg.TARGET_SIZE//2,3:].set(1.0)

    # CNN perception
    ident = jnp.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], dtype=jnp.float32)
    sobel_x = jnp.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], dtype=jnp.float32)/8
    lap = jnp.array([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]], dtype=jnp.float32)
    kernels = jnp.stack([ident, sobel_x, sobel_x.T, lap], axis=0)

    # make dummy tensor to init params
    dummy = jnp.zeros((1,cfg.TARGET_SIZE,cfg.TARGET_SIZE,cfg.TARGET_CH))
    dummy.at[:1,...,:4].set(target_img) 
    dummy = jnp.array(dummy, 'float32')

    # CNN wih perception
    class CA3(nn.Module):
        @nn.compact
        def __call__(self, img):
            x, alive_mask = self.perception_and_reshape(img, kernels)
            x = nn.Conv(features=128, kernel_size=(3,3), kernel_init=nn.initializers.glorot_uniform(), bias_init=nn.initializers.zeros, padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=cfg.TARGET_CH, kernel_size=(1,1), kernel_init=nn.initializers.zeros, use_bias=False)(x) 
            x = img + x
            x *= alive_mask
            return x

        def perception_and_reshape(self, x, kernels):
            n_chan = x.shape[-1] 
            n_kern = len(kernels)
            pre_life_mask = get_living_mask(x)
            x = perception(x, kernels)
            x = jnp.swapaxes(x, 1,2) 
            x = x.reshape((len(x),n_chan*n_kern,x.shape[-2],x.shape[-1]))
            x = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)))
            x = jnp.moveaxis(x, 1, -1)
            return x, pre_life_mask

    key = jax.random.PRNGKey(0)
    ca=CA3()
    params = ca.init(key, pool[:1,...])['params']
    out = ca.apply({'params': params}, dummy)

    # apply model, get the loss and backpropagate
    @jax.jit
    def apply_model(state, img):
        """Computes gradients, loss and accuracy for a single batch."""
        def loss_fn(params, x):
            def call_ca(_, x): return ca.apply({'params': params}, x)
            x = lax.fori_loop(0, 50, call_ca, x)
            loss = jnp.mean(jnp.square(x[...,:4]-target_img[...,:4]))
            return loss, x

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, x), grads = grad_fn(state.params, img)
        grads = jax.tree_map(lambda x: x/(jnp.linalg.norm(x)+1e-8), grads)
        state = state.apply_gradients(grads=grads)
        return state, grads, loss, x

    # init params and creat state
    key = jax.random.PRNGKey(0)
    ca=CA3()
    params_init = ca.init(key, pool[:1,...])['params']
    state = train_state.TrainState.create(
        apply_fn = ca.apply,
        params = params_init,
        tx = optax.adam(cfg.LR_INIT))

    # main training loop
    LR = cfg.LR_INIT
    key = random.PRNGKey(0)
    losses = []
    sks0 = random.split(key, cfg.EPOCHS)
    for idx_epochs, sk0 in tqdm(enumerate(sks0), total=len(sks0)):
        
        idx_batch = random.choice(sk0, jnp.arange(0, len(pool), 1, dtype=int), (1, cfg.BATCH_SIZE), replace=False)[0]
        x0 = pool[idx_batch]
        x0 = x0.at[:1].set(seed)
        # x0=seed
        state, grads, loss, x = apply_model(state, x0)
        losses.append(loss)

        pool = pool.at[idx_batch].set(x)

        if idx_epochs == cfg.EPOCHS//2:
            LR = LR * 0.1
            state = train_state.TrainState.create(
            apply_fn = ca.apply,
            params = state.params,
            tx = optax.adam(LR)) 

    # make synthesis figure
    def to_rgb(x):
        rgb, alpha = x[...,:3], jnp.clip(x[...,3:4],0,1)
        return 1-alpha+rgb

    x = seed
    imgs_syn = []
    with VideoWriter(filename = 'synthesis.mp4', fps=10.0) as vid:
        for i in tqdm(range(100)):
            im = jnp.clip(to_rgb(x)[0],0,1)
            vid.add(im)
            imgs_syn.append(im)
            x = ca.apply({'params': state.params}, x)
    fig_loss_and_synthesis(imgs_syn, losses, save='image_synthesis.png',
                            label=cfg.fig_label)

if __name__ == "__main__":
    main()