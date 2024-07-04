import io
import concurrent.futures
from multiprocessing import shared_memory

import numpy as np


class Closing:

  def __init__(self):
    self.closed = False

  def __enter__(self):
    # assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()
    self.closed = True


class SharedBuffer:

  ENABLE = True

  def __init__(self, content):
    if self.ENABLE:
      self.sm = shared_memory.SharedMemory(create=True, size=len(content))
      self.sm.buf[:] = memoryview(content)
      self.buf = self.sm.buf
    else:
      self.buf = bytes(content)

  def __getitem__(self, index):
    return self.buf[index]

  def __getstate__(self):
    if self.ENABLE:
      return self.sm.name
    else:
      return self.buf

  def __setstate__(self, value):
    if self.ENABLE:
      self.sm = shared_memory.SharedMemory(name=value)
      self.buf = self.sm.buf
    else:
      self.buf = value

  def open(self, mode='rb'):
    assert mode in ('rb', 'wb'), mode
    return io.BytesIO(self.buf)


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

  def close(self):
    self.sm.close()

  def unlink(self):
    self.sm.unlink()


class ThreadPool(concurrent.futures.ThreadPoolExecutor):

  def __init__(self, workers, name=None):
    super().__init__(workers, name)

  def submit(self, fn, *args, **kwargs):
    future = super().submit(fn, *args, **kwargs)
    # Prevent deamon threads from hanging due to exit handlers registered by
    # the concurrent.futures modules.
    concurrent.futures.thread._threads_queues.clear()
    return future

  def close(self, wait=False):
    self.shutdown(wait=wait)
