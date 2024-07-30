import atexit
import concurrent.futures
import collections
import functools
import math
import multiprocessing
import queue
import sys
import time
import traceback
import weakref
from multiprocessing import shared_memory

import numpy as np
import cloudpickle


class Loader:

  def __init__(
      self, source, batch, fns=(), shuffle=True,
      prefetch=10, workers=32, recycle_after=False,
      shard_id=0, num_shards=1, mp=None, seed=0):

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
    self.futures = collections.deque()
    self.batches = collections.deque()
    self.recycle_after = int(recycle_after)
    self.recycle_queue = collections.deque()

    self.mp = mp or multiprocessing.get_context('spawn')
    self.started = False
    self.stop = self.mp.Event()
    self.iqueue = self.mp.Queue()
    self.oqueue = self.mp.Queue()
    self.received = set()

    self.workers = []
    fns = cloudpickle.dumps(fns)
    args = (self.stop, self.iqueue, self.oqueue, source, fns, seed)
    for _ in range(workers):
      self.workers.append(self.mp.Process(target=self._worker, args=args))
    atexit.register(self.close)

  @functools.cached_property
  def spec(self):
    datapoint = self.source[0]
    datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
    for fn in self.fns:
      datapoint = fn(datapoint, seed=[0, 0])
      assert isinstance(datapoint, dict), fn
    return {k: (v.dtype, v.shape) for k, v in datapoint.items()}

  def __iter__(self):
    self.started = True
    for _ in range(self.prefetch):
      self._request()
    with concurrent.futures.ThreadPoolExecutor(len(self.workers)) as pool:
      list(pool.map(lambda x: x.start(), self.workers))
    return self

  def __next__(self):
    assert self.started
    try:
      self._request()
      batch = self._receive()
    except (SystemExit, KeyboardInterrupt):
      self.close()
      raise
    return batch

  def save(self):
    return {'step': self.consumed, 'seed': self.seed}

  def load(self, d):
    if self.started:
      for _ in range(self.prefetch):
        self._receive()
    self.consumed = self.step = d['step']
    self.seed = d['seed']
    for _ in range(self.prefetch):
      self._request()

  def close(self):
    self.stop.set()
    time.sleep(0.2)
    [x.join(timeout=0) for x in self.workers]
    [x.terminate() for x in self.workers if x.is_alive()]
    for q in (self.iqueue, self.oqueue):
      q.close()
      q.cancel_join_thread()
      q.join_thread()
    for batch in self.batches:
      [x.close() for x in batch.values()]
    self.batches.clear()

  @classmethod
  def _worker(cls, stop, iqueue, oqueue, source, fns, seed):
    try:
      fns = cloudpickle.loads(fns)
      while not stop.is_set():
        try:
          job = iqueue.get(timeout=0.1)
        except queue.Empty:
          continue
        index, step, batch, loc = job
        datapoint = source[index]
        datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
        for fn in fns:
          datapoint = fn(datapoint, seed=[seed, step])
          assert isinstance(datapoint, dict), fn
        assert datapoint.keys() == batch.keys()
        for key, value in datapoint.items():
          batch[key].array[loc] = value
        oqueue.put(step)
    except (SystemExit, KeyboardInterrupt):
      print('System exit in worker process.', flush=True)
      stop.set()
    except Exception:
      tb = ''.join(traceback.format_exception(sys.exception()))
      print('Exception in worker process:\n' + tb, flush=True)
      oqueue.put(tb)
      stop.set()

  def _request(self):
    if len(self.recycle_queue) > self.recycle_after:
      batch = self.recycle_queue.popleft()
    else:
      batch = {
          k: SharedArray((self.batch, *s), d)
          for k, (d, s) in self.spec.items()}
    self.batches.append(batch)
    for loc in range(self.batch):
      epoch = self.step // self.length
      index = self._order(epoch, self.seed)[self.step % self.length]
      self.iqueue.put((index, self.step, batch, loc))
      self.step += self.num_shards

  def _receive(self):
    collected = 0
    while collected < self.batch:
      try:
        result = self.oqueue.get(timeout=0.1)
        if not isinstance(result, int):
          self.close()
          raise RuntimeError(result)
        self.received.add(result)
      except queue.Empty:
        pass
      needed = self.consumed + collected
      if needed in self.received:
        self.received.remove(needed)
        collected += 1
    batch = self.batches.popleft()
    if self.recycle_after:
      self.recycle_queue.append(batch)
      batch = {k: v.array for k, v in batch.items()}
    else:
      batch = {k: v.result() for k, v in batch.items()}
    self.consumed += self.batch * self.num_shards
    return batch

  @functools.lru_cache(maxsize=1)
  def _order(self, epoch, seed):
    if self.shuffle:
      rng = np.random.default_rng(seed=[seed, epoch])
      return rng.permutation(np.arange(self.length)).tolist()
    else:
      return list(range(self.length))


class SharedArray:

  def __init__(self, shape, dtype, name=None):
    if name:
      self.shm = shared_memory.SharedMemory(name=name)
    else:
      size = math.prod(shape) * np.dtype(dtype).itemsize
      self.shm = shared_memory.SharedMemory(create=True, size=size)
    self.arr = np.ndarray(shape, dtype, self.shm.buf)
    weakref.finalize(self.arr, self.shm.close)

  @property
  def array(self):
    return self.arr

  def result(self):
    weakref.finalize(self.arr, self.close)
    return self.arr

  def close(self):
    self.arr = None
    try:
      self.shm.unlink()
    except FileNotFoundError:
      pass

  def __getstate__(self):
    return (self.arr.shape, self.arr.dtype.str, self.shm.name)

  def __setstate__(self, args):
    self.__init__(*args)
