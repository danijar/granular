import functools

import numpy as np


def convert(source):
  if callable(source):
    return source
  elif hasattr(source, '__getitem__'):
    return source.__getitem__
  else:
    raise TypeError(source)


class Transform:

  def __init__(self, source, fn, seed=0):
    self.source = convert(source)
    self.fn = fn
    self.seed = seed

  def __call__(self, step):
    datapoint = self.source(step)
    datapoint = self.fn(datapoint, seed=[self.seed, step])
    return datapoint


class Sample:

  def __init__(self, source, length=None, seed=0):
    length = length or len(source)
    source = convert(source)
    self.source = source
    self.length = length
    self.seed = seed

  def __call__(self, step):
    rng = np.random.default_rng(seed=[self.seed, step])
    index = rng.choice(self.length)
    return self.source(index)


class Epochs:

  def __init__(self, source, length=None, shuffle=True, seed=0):
    length = length or len(source)
    source = convert(source)
    self.source = source
    self.length = length
    self.shuffle = shuffle
    self.seed = 0

  def __call__(self, step):
    epoch = step // self.length
    index = step % self.length
    if self.shuffle:
      index = self._permutation(epoch)[index]
    return self.source(index)

  @functools.lru_cache(maxsize=1)
  def _permutation(self, epoch):
    rng = np.random.default_rng(seed=[self.seed, epoch])
    return rng.permutation(np.arange(self.length)).tolist()


class Truncate:

  def __init__(self, source, limit, length=None):
    length = length or len(source)
    self.source = convert(source)
    self.limit = min(length, limit)

  def __call__(self, step):
    return self.source(step % self.limit)


class Interleave:

  def __init__(self, sources):
    self.sources = [convert(source) for source in sources]

  def __call__(self, step):
    outer = step % len(self.sources)
    inner = step // len(self.sources)
    return self.sources[outer](inner)


class Mix:

  def __init__(self, sources, weights, seed=0):
    self.sources = [convert(source) for source in sources]
    weights = np.asarray(weights, np.float32)
    weights /= weights.sum()
    self.weights = weights
    self.seed = seed

  def __call__(self, step):
    rng = np.random.default_rng(seed=[self.seed, step])
    choice = rng.choice(len(self.sources), p=self.weights)
    return self.sources[choice](step)
