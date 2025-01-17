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


class Loader:

  def __init__(
      self, source, batch, fns=(), shuffle=None,
      prefetch=8, workers=32, recycle_after=False,
      shard_id=0, num_shards=1, mp=None):

    assert shuffle is None, (
        'Shuffling has been removed from the Loader. Shuffle the source that '
        'before passing it into the Loader instead:\n'
        'source = granular.sources.Epochs(source, shuffle=True, seed=0)')
    del shuffle

    assert fns == (), (
        'Transformations have been removed from the Loader. Transform the '
        'source before passing it into the loader instead:\n'
        'source = granular.sources.Transform(source, fn=..., seed=0).')
    del fns

    self.source = source
    self.batch = batch
    self.prefetch = prefetch
    self.shard_id = shard_id
    self.num_shards = num_shards

    self.step = 0
    self.consumed = 0
    self.futures = collections.deque()
    self.batches = collections.deque()
    self.recycle_after = recycle_after
    self.recycle_queue = collections.deque()

    self.mp = mp or multiprocessing.get_context('spawn')
    self.started = False
    self.stop = self.mp.Event()
    self.iqueue = self.mp.Queue()
    self.oqueue = self.mp.Queue()
    self.received = set()

    self.workers = []
    args = (self.stop, self.iqueue, self.oqueue, source)
    for _ in range(workers):
      self.workers.append(self.mp.Process(target=self._worker, args=args))
    atexit.register(self.close)

  @functools.cached_property
  def spec(self):
    datapoint = self.source(0)
    datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
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
    return {'step': self.consumed}

  def load(self, d):
    if self.started:
      for _ in range(self.prefetch):
        self._receive()
    self.consumed = self.step = d['step']
    if self.started:
      for _ in range(self.prefetch):
        self._request()

  def close(self):
    self.stop.set()
    time.sleep(0.2)
    if self.started:
      for worker in self.workers:
        try:
          worker.join(timeout=0)
        except AssertionError:
          pass
      [x.terminate() for x in self.workers if x.is_alive()]
    for q in (self.iqueue, self.oqueue):
      q.close()
      q.cancel_join_thread()
      q.join_thread()
    for batch in list(self.batches) + list(self.recycle_queue):
      for buffer in batch.values():
        buffer.close()
    self.batches.clear()
    self.recycle_queue.clear()

  @classmethod
  def _worker(cls, stop, iqueue, oqueue, source):
    try:
      while not stop.is_set():
        try:
          job = iqueue.get(timeout=0.1)
        except queue.Empty:
          continue
        step, batch, loc = job
        datapoint = source(step)
        assert isinstance(datapoint, dict)
        datapoint = {k: np.asarray(v) for k, v in datapoint.items()}
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
    if self.recycle_after and len(self.recycle_queue) > self.recycle_after:
      batch = self.recycle_queue.popleft()
    else:
      batch = {
          k: SharedArray((self.batch, *s), d)
          for k, (d, s) in self.spec.items()}
    self.batches.append(batch)
    for loc in range(self.batch):
      local_step = self.step + self.shard_id * self.batch + loc
      self.iqueue.put((local_step, batch, loc))
    self.step += self.num_shards * self.batch

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
      needed = self.consumed + self.shard_id * self.batch + collected
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
    # We cannot use self.close as finalizer because that would keep self alive
    # and thus the reference count for self.shm would never reach zero.
    weakref.finalize(self.arr, self.shm.unlink)
    arr = self.arr
    self.arr = None  # Prevent future usage
    return arr

  def close(self):
    self.arr = None  # Prevent future usage
    self.shm.unlink()

  def __getstate__(self):
    return (self.arr.shape, self.arr.dtype.str, self.shm.name)

  def __setstate__(self, args):
    self.__init__(*args)


if __name__ == '__main__':
  import psutil
  process = psutil.Process()
  gb = 1024 ** 3
  source = [{'foo': np.zeros(gb, np.uint8)}]
  loader = Loader(
      source, batch=2, prefetch=2, workers=4,
      # recycle_after=2,
  )
  for i, batch in enumerate(loader):
    time.sleep(1)
    info = process.memory_info()
    print(
        f'step={i}',
        f'rss={info.rss / gb:.2f}',
        f'vms={info.vms / gb:.2f}',
        f'shm={info.shared / gb:.2f}',
    )
