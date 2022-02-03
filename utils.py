import jax
import jax.numpy as jnp
import PIL
import requests
import io
import matplotlib.pyplot as plt
import jax.scipy as jsp
from jax import lax
from flax import linen as nn
from jax import vmap
from tqdm.notebook import tqdm
import numpy as np

from flax.training import train_state
import optax
import ml_collections

import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def imread(url, max_size=None, mode=None):
    if url.startswith(('http:', 'https:')):
        # wikimedia requires a user agent
        headers = {
          "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    if mode is not None:
        img = img.convert(mode)
    img = jnp.array(img, dtype=jnp.float32)/255.0
    return img

def load_image(url, max_size=40):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    return load_image(url)

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()


class GetLivingMask(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.max_pool(x[...,3:4], (3,3), padding='SAME')
      return x > 0.1

def get_living_mask(x):
      x = nn.max_pool(x[...,3:4], (3,3), padding='SAME')
      return x > 0.1
