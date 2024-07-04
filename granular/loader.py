import collections
import concurrent.futures
import functools
from multiprocessing import shared_memory

import numpy as np
import cloudpickle


class Loader:

  def __init__(
      self, source, fns, batch, shuffle=True, prefetch=10,
      processes=64, shard_id=0, num_shards=1, seed=0):

    self.source = source
    self.fns = fns
    self.batch = batch
    self.shuffle = shuffle
    self.prefetch = prefetch
    self.shard_id = shard_id
    self.num_shards = num_shards
    self.seed = seed

    self.length = len(source)
    self.step = 0
    self.consumed = 0
    self.started = False
    self.batches = collections.deque()
    self.futures = collections.deque()

    fns = cloudpickle.dumps(fns)
    initargs = (source, fns, seed)
    self.pool = concurrent.futures.ProcessPoolExecutor(
        processes, None, self._init, initargs)

  @functools.cached_property
  def spec(self):
    datapoint = self.source[0]
    datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
    for fn in self.fns:
      datapoint = fn(datapoint, seed=[0, 0])
      assert isinstance(datapoint, dict), fn
    return {k: (v.dtype, v.shape) for k, v in datapoint.items()}

  def save(self):
    return {'step': self.consumed, 'seed': self.seed}

  def load(self, d):
    self.consumed = self.step = d['step']
    self.seed = d['seed']

  def __iter__(self):
    self.started = True
    for _ in range(self.prefetch):
      self._request()
    return self

  def __next__(self):
    assert self.started
    try:
      self._request()
      batch = self._receive()
      self.consumed += self.batch * self.num_shards
    except (Exception, KeyboardInterrupt, SystemExit):
      self.pool.shutdown(wait=False, cancel_futures=True)
      raise
    return batch

  @classmethod
  def _init(cls, source, fns, seed):
    fns = cloudpickle.loads(fns)
    globals()['initargs'] = (source, fns, seed)

  @classmethod
  def _load(cls, index, step, batchdesc, loc):
    source, fns, seed = globals()['initargs']
    batch = {k: SharedArray(*v) for k, v in batchdesc.items()}
    datapoint = source[index]
    datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
    for fn in fns:
      datapoint = fn(datapoint, seed=[seed, step])
      assert isinstance(datapoint, dict), fn
    assert datapoint.keys() == batch.keys()
    for key, value in datapoint.items():
      batch[key][loc] = value

  def _request(self):
    batch = {
        k: SharedArray(d, (self.batch, *s))
        for k, (d, s) in self.spec.items()}
    batchdesc = {k: v.desc for k, v in batch.items()}
    self.batches.append(batch)
    for loc in range(self.batch):
      epoch = self.step // self.length
      index = self._order(epoch)[self.step % self.length]
      args = (index, self.step, batchdesc, loc)
      self.futures.append(self.pool.submit(self._load, *args))
      self.step += self.num_shards

  def _receive(self):
    for _ in range(self.batch):
      self.futures.popleft().result()
    return self.batches.popleft()

  @functools.lru_cache(maxsize=1)
  def _order(self, epoch):
    if self.shuffle:
      rng = np.random.default_rng(seed=[self.seed, epoch])
      return rng.permutation(np.arange(self.length)).tolist()
    else:
      return list(range(self.length))


class SharedArray:

  def __init__(self, dtype, shape, name=None):
    dtype = np.dtype(dtype)
    self.dtype = dtype
    self.shape = shape
    size = int(np.prod(shape)) * dtype.itemsize
    if name:
      self.sm = shared_memory.SharedMemory(name=name, size=size)
      self.created = True
    else:
      self.sm = shared_memory.SharedMemory(create=True, size=size)
      self.created = False
    self.array = np.ndarray(shape, dtype, self.sm.buf)
    assert self.array.data.c_contiguous

  @property
  def name(self):
    return self.sm.name

  @property
  def desc(self):
    return (self.dtype.str, self.shape, self.name)

  def __getattr__(self, name):
    return getattr(self.array, name)

  def __getitem__(self, index):
    return self.array[index]

  def __setitem__(self, index, value):
    self.array[index] = value
