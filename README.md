# jax_cellular_automata
## Overview

## Results
![image_synthesis](figures/image_synthesis_withPerc.png?raw=true) 

## Perception
We can have a vectorized version of perception (channel wise convolution with specific filters) using vmap. In this case we apply vmap on each axis and convolve each channel of the input image by our identity, sobel and laplacian filters.

```python
ident = jnp.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], dtype=jnp.float32)
sobel_x = jnp.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], dtype=jnp.float32)/8
lap = jnp.array([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]], dtype=jnp.float32)
kernels = jnp.stack([ident, sobel_x, sobel_x.T, lap], axis=0)

@jax.jit
def perception(img, kernels): 
    _perception = vmap(vmap(vmap(conv2d, in_axes=(-1, None)), in_axes=(None,0)), in_axes=(0, None)) 
    return _perception(img, kernels)
```

![image_synthesis](figures/perception_jax.png?raw=true) 