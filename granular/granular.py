import concurrent.futures
import functools
import io
import itertools
import json
import operator
import os
import pathlib
import pickle
import re
from multiprocessing import shared_memory

import msgpack


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


class ShardedDatasetWriter(Closing):

  def __init__(
      self, directory, spec, encoders,
      shardlen=None, shardstart=0, shardstep=1):
    super().__init__()
    assert 0 <= shardstart
    assert 1 <= shardstep
    if shardstart > 0 or shardstep > 1:
      assert shardlen, shardlen
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    try:
      directory.mkdir()
    except FileExistsError:
      pass
    self.directory = directory
    self.thespec = spec
    self.encoders = encoders
    self.shardlength = shardlen
    self.shardstart = shardstart
    self.shardnum = shardstart
    self.shardstep = shardstep
    self.prevshards = 0
    self.prevsize = 0
    self.prevlength = 0
    self.writer = None

  @property
  def spec(self):
    return self.thespec

  @property
  def shards(self):
    return self.prevshards + bool(self.writer)

  @property
  def size(self):
    return self.prevsize + (self.writer.size if self.writer else 0)

  def __len__(self):
    return self.prevlength + (len(self.writer) if self.writer else 0)

  def append(self, datapoint, flush=True):
    if not self.writer:
      folder = self.directory / f'{self.shardnum:06}'
      self.writer = DatasetWriter(folder, self.spec, self.encoders)
    self.writer.append(datapoint)
    if self.shardlength and len(self.writer) >= self.shardlength:
      self.prevshards += 1
      self.prevsize += self.writer.size
      self.prevlength += len(self.writer)
      self.writer.close()
      self.writer = None
      self.shardnum += self.shardstep
    return len(self) - 1

  def flush(self):
    if not self.specwritten:
      self.specwritten = True
      content = json.dumps(self.spec).encode('utf-8')
      with (self.directory / 'spec.json').open('wb') as f:
        f.write(content)
    self.refwriter.flush()
    for writer in self.writers.values():
      writer.flush()

  def close(self):
    if self.writer:
      self.writer.close()


class ShardedDatasetReader(Closing):

  def __init__(
      self, directory, decoders, cache_index=True, cache_keys=(),
      parallel=False, shardstart=0, shardstep=1):
    super().__init__()
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    folders = sorted(directory.glob('*'))
    assert all(int(x.name) == i for i, x in enumerate(folders)), folders
    selected = [
        folders[i] for i in range(shardstart, len(folders), shardstep)]
    assert selected, (folders, selected, shardstart, shardstep)
    if parallel:
      make = lambda x: DatasetReader(
          x, decoders, cache_index, cache_keys, parallel)
      with concurrent.futures.ThreadPoolExecutor(len(selected)) as pool:
        self.readers = list(pool.map(make, selected))
    else:
      self.readers = [
          DatasetReader(x, decoders, cache_index, cache_keys, parallel)
          for x in selected]
    lengths = [len(x) for x in self.readers]
    self.stops = list(itertools.accumulate(lengths, operator.add))
    self.starts = [0] + self.stops[:-1]
    self.length = sum(lengths)

  @property
  def spec(self):
    return self.readers[0].spec

  @property
  def size(self):
    return sum(x.size for x in self.readers)

  @property
  def shards(self):
    return len(self.readers)

  def mask(self, index):
    reader, local_index = self._resolve(index)
    return reader.mask(local_index)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index, mask = index
      reader, local_index = self._resolve(index)
      return reader[local_index, mask]
    else:
      reader, local_index = self._resolve(index)
      return reader[local_index]

  def copy(self):
    return pickle.loads(pickle.dumps(self))

  def close(self):
    for reader in self.readers:
      reader.close()

  def _resolve(self, index):
    if not (0 <= index <= self.length):
      raise IndexError(index)
    for reader, start, stop in zip(self.readers, self.starts, self.stops):
      if start <= index < stop:
        local_index = index - start
        return reader, local_index


class DatasetWriter(Closing):

  def __init__(self, directory, spec, encoders):
    super().__init__()
    assert all(re.match(r'^[a-z_]', k) for k in spec.keys()), spec
    assert all(re.match(r'^[a-z_](\[\])?', v) for v in spec.values()), spec
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    assert not directory.exists(), directory
    directory.mkdir()
    spec = dict(sorted(spec.items(), key=lambda x: x[0]))
    if encoders is None:
      encoders = {k: None for k in spec.keys()}
    else:
      encoders = {k: encoders[v.rstrip('[]')] for k, v in spec.items()}
    self.directory = directory
    self.encoders = encoders
    self.thespec = spec
    self.refwriter = BagWriter(self.directory / 'refs.bag')
    self.writers = {
        k: BagWriter(self.directory / f'{k}.bag')
        for k in self.spec.keys()}
    self.specwritten = False

  @property
  def spec(self):
    return self.thespec

  @property
  def size(self):
    return sum(x.size for x in self.writers.values()) + self.refwriter.size

  def __len__(self):
    return len(self.refwriter)

  def append(self, datapoint, flush=True):
    assert isinstance(datapoint, dict)
    assert set(datapoint.keys()) == set(self.spec.keys()), (
        datapoint.keys(), self.spec)
    refs = []
    # Iterate in sorted key order.
    for key, dtype in self.spec.items():
      writer = self.writers[key]
      if dtype.endswith('[]'):
        # Hold only one value of the generator in memory at a time.
        indices = []
        for value in datapoint[key]:
          value = self._encode(key, value, dtype)
          index = writer.append(value, flush=False)
          indices.append(index)
        if indices:
          refs.append([indices[0], len(indices)])
        else:
          refs.append([])
      else:
        value = datapoint[key]
        value = self._encode(key, value, dtype)
        index = writer.append(value, flush=False)
        refs.append(index)
    refs = msgpack.packb(refs)
    index = self.refwriter.append(refs, flush=False)
    flush and self.flush()
    return index

  def flush(self):
    if not self.specwritten:
      self.specwritten = True
      content = json.dumps(self.spec).encode('utf-8')
      with (self.directory / 'spec.json').open('wb') as f:
        f.write(content)
    self.refwriter.flush()
    for writer in self.writers.values():
      writer.flush()

  def close(self):
    self.refwriter.close()
    for writer in self.writers.values():
      writer.close()

  def _encode(self, key, value, dtype):
    encoder = self.encoders[key]
    if not encoder:
      return value
    try:
      value = encoder(value)
      assert isinstance(value, bytes), (key, type(value))
      return value
    except Exception:
      print(f"Error encoding key '{key}' of type '{dtype}'.")
      raise


class DatasetReader(Closing):

  def __init__(
      self, directory, decoders,
      cache_index=True, cache_keys=(), parallel=False):
    super().__init__()
    assert isinstance(cache_keys, (tuple, list)), cache_keys
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    with (directory / 'spec.json').open('rb') as f:
      self.thespec = json.loads(f.read())
    assert all(k in self.thespec or k == 'refs' for k in cache_keys), (
        self.thespec, cache_keys)
    self.readers = {
        k: BagReader(
            directory / f'{k}.bag', cache_index, cache_data=(k in cache_keys))
        for k in ('refs', *self.spec.keys())}
    if decoders is None:
      decoders = {k: None for k in self.spec.keys()}
    else:
      decoders = {k: decoders[v.rstrip('[]')] for k, v in self.spec.items()}
    self.decoders = decoders
    self.parallel = parallel
    if parallel:
      self.pool = self._make_pool()

  @property
  def spec(self):
    return self.thespec

  @property
  def size(self):
    return sum(x.size for x in self.readers.values())

  def mask(self, index):
    return {
        key: range(0, ref[1] if ref else 0) if dtype.endswith('[]') else True
        for ref, (key, dtype) in zip(self._getref(index), self.spec.items())}

  def __getstate__(self):
    d = self.__dict__.copy()
    if self.parallel:
      d.pop('pool', None)
    return d

  def __setstate__(self, d):
    self.__dict__.update(d)
    if self.parallel:
      self.pool = self._make_pool()

  def __len__(self):
    return len(self.readers['refs'])

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index, mask = index
      assert isinstance(mask, dict), mask
    else:
      mask = {k: True for k in self.spec.keys()}
    assert all(k in self.spec for k in mask), (self.spec, mask)
    ref = self._getref(index)
    requests = {}
    for i, (key, dtype) in enumerate(self.spec.items()):
      msk = mask.get(key, False)
      if dtype.endswith('[]'):
        if msk is False:
          continue
        if ref[i]:
          start, length = ref[i]
          avail = range(start, start + length)
        else:
          avail = range(0)
        if isinstance(msk, bool):
          assert msk is True
          requests[key] = avail
        elif isinstance(msk, (slice, range)):
          assert msk.start >= 0 and msk.stop >= 0 and msk.step == 1, msk
          start = avail.start + msk.start
          stop = min(avail.start + msk.stop, avail.stop)
          requests[key] = range(start, stop)
        else:
          raise TypeError((msk, type(msk)))
      else:
        if not isinstance(msk, bool):
          raise TypeError((key, mask))
        if not msk:
          continue
        requests[key] = ref[i]
    if self.parallel:
      values = self.pool.map(
          lambda it: self.readers[it[0]][it[1]], requests.items())
      datapoint = dict(zip(requests.keys(), values))
    else:
      datapoint = {k: self.readers[k][r] for k, r in requests.items()}
    decoded = {
        k: self._decode(k, v, self.spec[k])
        for k, v in datapoint.items()}
    return decoded

  def copy(self):
    return pickle.loads(pickle.dumps(self))

  def close(self):
    for reader in self.readers.values():
      reader.close()

  @functools.lru_cache(maxsize=1)
  def _getref(self, index):
    ref = self.readers['refs'][index]
    ref = msgpack.unpackb(ref)
    assert len(ref) == len(self.spec)
    for (key, dtype), r in zip(self.spec.items(), ref):
      if dtype.endswith('[]'):
        assert isinstance(r, list) and len(r) in (0, 2)
      else:
        assert isinstance(r, int)
    return ref

  def _decode(self, key, value, dtype):
    decoder = self.decoders[key]
    if not decoder:
      return value
    try:
      if isinstance(value, list):
        return [decoder(x) for x in value]
      else:
        return decoder(value)
    except Exception:
      print(f"Error decoding key '{key}' of type '{dtype}'.")
      raise

  def _make_pool(self):
    return concurrent.futures.ThreadPoolExecutor(len(self.spec))


class BagWriter(Closing):

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


class BagReader(Closing):

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
    self.file = source.open('rb')
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
    self.file = self.source.open('rb')

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
    self.file.close()

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
