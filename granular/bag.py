import io
import os
import pathlib
import pickle
from multiprocessing import shared_memory

from . import utils


class BagWriter(utils.Closing):

  def __init__(self, filepath):
    super().__init__()
    if isinstance(filepath, str):
      filepath = pathlib.Path(filepath)
    assert not filepath.exists(), filepath
    file = filepath.open('wb')
    assert isinstance(file, io.BufferedIOBase), type(file)
    assert file.writable(), file
    self.file = file
    self.offset = 0
    self.length = 0
    self.limits = []
    self.towrite = []

  @property
  def size(self):
    return self.offset + 8 * self.length + 8

  def __len__(self):
    return self.length

  def append(self, record, flush=True):
    assert not self.closed
    assert self.length < 2 ** 32 - 1, self.length
    assert isinstance(record, bytes), type(record)
    index = self.length
    self.length += 1
    self.limits.append(self.offset)
    self.offset += len(record)
    self.towrite.append(record)
    flush and self.flush()
    return index

  def flush(self):
    if not self.towrite:
      return
    if len(self.towrite) == 1:
      self.file.write(self.towrite[0])
    else:
      self.file.write(b''.join(self.towrite))
    self.towrite.clear()

  def close(self):
    assert not self.closed
    self.limits.append(self.offset)
    self.towrite.extend(tuple(x.to_bytes(8, 'little') for x in self.limits))
    self.flush()
    self.file.close()


class BagReader(utils.Closing):

  def __init__(self, source, cache_index=True, cache_data=False):
    super().__init__()
    if isinstance(source, str):
      source = pathlib.Path(source)
    if cache_data and hasattr(source, 'open'):
      cache_index = False
      with source.open('rb') as f:
        source = SharedBuffer(f.read())
    if not hasattr(source, 'open'):
      cache_index, cache_data = False, False
      source = SharedBuffer(source)
    self.source = source
    self.fp = None
    assert self.file.readable(), self.file
    self.file.seek(-8, os.SEEK_END)
    self.filesize = self.file.tell() + 8
    self.recordend = int.from_bytes(self.file.read(8), 'little')
    self.length = (self.filesize - 8 - self.recordend) // 8
    if cache_index:
      self.file.seek(self.recordend, os.SEEK_SET)
      limits = self.file.read(8 * self.length + 8)
      self.limits = SharedBuffer(limits)
    else:
      self.limits = None

  @property
  def file(self):
    if not self.fp:
      self.fp = self.source.open('rb')
    return self.fp

  def __getstate__(self):
    return {
        'source': self.source,
        'filesize': self.filesize,
        'recordend': self.recordend,
        'length': self.length,
        'limits': self.limits,
    }

  def __setstate__(self, d):
    self.source = d['source']
    self.filesize = d['filesize']
    self.recordend = d['recordend']
    self.length = d['length']
    self.limits = d['limits']
    self.closed = False
    self.fp = None

  @property
  def size(self):
    return self.filesize

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert not self.closed
    assert isinstance(index, (int, slice, range)), (index, type(index))
    if isinstance(index, int):
      start = self._get_start(index)
      if index + 1 < self.length:
        end = self._get_start(index + 1)
      else:
        end = self.recordend
      self.file.seek(start, os.SEEK_SET)
      return self.file.read(end - start)
    else:
      assert index.start >= 0 and index.stop >= 0 and index.step == 1, index
      index = range(index.start, min(index.stop, self.length))
      if index.start == index.stop:
        return []
      limits = self._get_limits(index.start, index.stop + 1)
      start, stop = limits[0], limits[-1]
      self.file.seek(start, os.SEEK_SET)
      buffer = self.file.read(stop - start)
      records = [
          buffer[i - start: j - start]
          for i, j in zip(limits[:-1], limits[1:])]
      return records

  def copy(self):
    return pickle.loads(pickle.dumps(self))

  def close(self):
    assert not self.closed
    if self.fp:
      self.fp.close()
    if isinstance(self.limits, SharedBuffer):
      self.limits.close()
    if isinstance(self.source, SharedBuffer):
      self.source.close()

  def _get_start(self, index):
    if self.limits:
      return int.from_bytes(self.limits[8 * index: 8 * (index + 1)], 'little')
    else:
      self.file.seek(-8 * (self.length - index) - 8, os.SEEK_END)
      offset = self.file.read(8)
      offset = int.from_bytes(offset, 'little')
      return offset

  def _get_limits(self, start, stop):
    if self.limits:
      return [
          int.from_bytes(self.limits[8 * i: 8 * (i + 1)], 'little')
          for i in range(start, stop)]
    else:
      self.file.seek(-8 * (self.length - start) - 8, os.SEEK_END)
      limits = self.file.read(8 * (stop - start))
      limits = [
          int.from_bytes(limits[8 * i: 8 * (i + 1)], 'little')
          for i in range(stop - start)]
      return limits


class SharedBuffer:

  ENABLE = True

  def __init__(self, content):
    if self.ENABLE:
      self.shm = shared_memory.SharedMemory(create=True, size=len(content))
      self.shm.buf[:] = memoryview(content)
      self.buf = self.shm.buf
    else:
      self.buf = bytes(content)

  def __getitem__(self, index):
    return self.buf[index]

  def __getstate__(self):
    if self.ENABLE:
      return self.shm.name
    else:
      return self.buf

  def __setstate__(self, value):
    if self.ENABLE:
      self.shm = shared_memory.SharedMemory(name=value)
      self.buf = self.shm.buf
    else:
      self.buf = value

  def open(self, mode='rb'):
    assert mode in ('rb', 'wb'), mode
    return io.BytesIO(self.buf)

  def close(self):
    self.buf = None
    try:
      self.shm.unlink()
    except FileNotFoundError:
      pass
