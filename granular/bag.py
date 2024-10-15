import errno
import io
import os
import pathlib
import pickle
import struct
from multiprocessing import shared_memory

from . import utils


limst = struct.Struct('<Q')


class BagWriter(utils.Closing):

  def __init__(self, filepath, version=2):
    assert version in (1, 2), version
    super().__init__()
    if isinstance(filepath, str):
      filepath = pathlib.Path(filepath)
    assert not filepath.exists(), filepath
    file = filepath.open('wb')
    assert isinstance(file, io.BufferedIOBase), type(file)
    assert file.writable(), file
    # Version 1 stores the starts and ends of all blobs, resulting in N+1
    # entries in the limits table, where the first entry is zero.
    # Version 2 stores only the ends of all blobs, resulting in N entries in
    # the limits table.
    self.version = version
    self.file = file
    self.offset = 0
    self.length = 0
    self.limits = []
    self.towrite = []

  @property
  def size(self):
    return self.offset + 8 * self.length + (8 if self.version == 1 else 0)

  def __len__(self):
    return self.length

  def append(self, record, flush=True):
    assert len(record), 'zero byte record'
    assert not self.closed
    assert self.length < 2 ** 32 - 1, self.length
    assert isinstance(record, bytes), type(record)
    index = self.length
    self.length += 1
    if self.version == 1:
      self.limits.append(self.offset)
      self.offset += len(record)
    if self.version == 2:
      self.offset += len(record)
      self.limits.append(self.offset)
    self.towrite.append(record)
    flush and self.flush()
    return index

  def flush(self):
    assert not self.closed
    if not self.towrite:
      return
    if len(self.towrite) == 1:
      self.file.write(self.towrite[0])
    else:
      self.file.write(b''.join(self.towrite))
    self.towrite.clear()

  def close(self):
    assert not self.closed
    if self.version == 1:
      self.limits.append(self.offset)
    self.towrite.extend(limst.pack(x) for x in self.limits)
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
    try:
      self.file.seek(-8, os.SEEK_END)
      empty = False
    except OSError as e:
      if e.errno != errno.EINVAL:
        raise
      empty = True
    if empty:
      self.filesize = 0
      self.indexstart = 0
      self.length = 0
    else:
      self.filesize = self.file.tell() + 8
      self.indexstart = limst.unpack(self.file.read(8))[0]
      # If the index table starts with a zero, we are reading a version 1 file
      # and we can skip the initial entry.
      self.file.seek(self.indexstart, os.SEEK_SET)
      if limst.unpack(self.file.read(8))[0] == 0:
        self.indexstart += 8
      self.length = (self.filesize - self.indexstart) // 8
    if cache_index and self.length:
      self.file.seek(self.indexstart, os.SEEK_SET)
      limits = self.file.read(self.filesize - self.indexstart)
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
        'indexstart': self.indexstart,
        'length': self.length,
        'limits': self.limits,
    }

  def __setstate__(self, d):
    self.source = d['source']
    self.filesize = d['filesize']
    self.indexstart = d['indexstart']
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
      lhs, rhs = self._getlims(index, index + 1)
      self.file.seek(lhs, os.SEEK_SET)
      return self.file.read(rhs - lhs)
    else:
      assert 0 <= index.start <= index.stop and index.step in (None, 1), index
      index = range(index.start, min(index.stop, self.length))
      if index.start == index.stop:
        return []
      limits = self._getlims(index.start, index.stop)
      lhs, rhs = limits[0], limits[-1]
      self.file.seek(lhs, os.SEEK_SET)
      buffer = self.file.read(rhs - lhs)
      records = [
          buffer[i - lhs: j - lhs]
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

  def _getlims(self, start, stop):
    assert start < stop, (start, stop)
    lhs = 8 * max(0, start - 1)
    rhs = 8 * stop
    if self.limits:
      buffer = self.limits[lhs: rhs]
    else:
      self.file.seek(self.indexstart + lhs, os.SEEK_SET)
      buffer = self.file.read(rhs - lhs)
    limits = [
        limst.unpack(buffer[i: i + 8])[0]
        for i in range(0, len(buffer), 8)]
    return limits if start else (0, *limits)


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
