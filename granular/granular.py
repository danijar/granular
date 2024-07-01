import io
import itertools
import json
import operator
import os
import pathlib
import re

import msgpack


GB = 1024 ** 3


def parse_spec(spec):
  assert isinstance(spec, dict), spec
  assert 'refs' not in spec
  assert list(spec.keys()) == sorted(spec.keys())
  parsed = {}
  for key, value in spec.items():
    assert isinstance(key, str), key
    islist = value.endswith('[]')
    value = value[:-2] if islist else value
    mat = re.match(r'^([a-z0-9-_]+)(\([a-z0-9-_,.]*\))?$', value)
    assert mat, (key, value)
    dtype = mat.group(1)
    if mat.group(2):
      args = tuple(mat.group(2)[1:-1].split(','))
    else:
      args = ()
    parsed[key] = (islist, dtype, args)
  return parsed


class Closing:

  def __init__(self):
    self.closed = False

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()
    self.closed = True


class ShardedDatasetWriter(Closing):

  def __init__(
      self, directory, spec, encoders=None,
      shardlen=None,
      shardstart=0, shardstep=1):
    super().__init__()
    assert 0 <= shardstart
    assert 1 <= shardstep
    if shardstep > 1:
      print('NOTE: Writing with shard step cannot preserve the record order.')
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    try:
      directory.mkdir()
    except FileExistsError:
      pass
    self.directory = directory
    self.rawspec = spec
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
    return self.rawspec

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
      content = json.dumps(self.rawspec).encode('utf-8')
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
      self, directory, decoders=None, cache_index=True, cache_refs=False,
      shardstart=0, shardstep=1):
    super().__init__()
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    folders = sorted(directory.glob('*'))
    assert all(int(x.name) == i for i, x in enumerate(folders)), folders
    selected = [
        folders[i] for i in range(shardstart, len(folders), shardstep)]
    assert selected, (folders, selected, shardstart, shardstep)
    self.readers = [
        DatasetReader(x, decoders, cache_index, cache_refs) for x in selected]
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

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    if not (0 <= index <= self.length):
      raise IndexError(index)
    for reader, start, stop in zip(self.readers, self.starts, self.stops):
      if start <= index < stop:
        return reader[index - start]

  def close(self):
    for reader in self.readers:
      reader.close()


class DatasetWriter(Closing):

  def __init__(self, directory, spec, encoders=None):
    super().__init__()
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    assert not directory.exists(), directory
    directory.mkdir()
    spec = dict(sorted(spec.items(), key=lambda x: x[0]))
    self.directory = directory
    self.encoders = encoders
    self.rawspec = spec
    self.parsedspec = parse_spec(spec)
    self.refwriter = BagWriter(self.directory / 'refs.bag')
    self.writers = {
        k: BagWriter(self.directory / f'{k}.bag')
        for k in self.parsedspec.keys()}
    self.specwritten = False

  @property
  def spec(self):
    return self.rawspec

  @property
  def size(self):
    return sum(x.size for x in self.writers.values()) + self.refwriter.size

  def __len__(self):
    return len(self.refwriter)

  def append(self, datapoint, flush=True):
    assert isinstance(datapoint, dict)
    assert set(datapoint.keys()) == set(self.parsedspec.keys()), (
        datapoint.keys(), self.parsedspec)
    refs = []
    # Iterate in sorted key order.
    for key, (islist, dtype, args) in self.parsedspec.items():
      writer = self.writers[key]
      if islist:
        # Hold only one value of the generator in memory at a time.
        indices = []
        for value in datapoint[key]:
          if self.encoders is not None:
            value = self.encoders[dtype](value, *args)
          assert isinstance(value, bytes), (key, type(value))
          index = writer.append(value, flush=False)
          indices.append(index)
        if indices:
          refs.append([indices[0], len(indices)])
        else:
          refs.append([])
      else:
        value = datapoint[key]
        if self.encoders is not None:
          value = self.encoders[dtype](value, *args)
        assert isinstance(value, bytes), (key, type(value))
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


class DatasetReader(Closing):

  def __init__(
      self, directory, decoders=None, cache_index=True, cache_refs=False):
    super().__init__()
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    with (directory / 'spec.json').open('rb') as f:
      self.rawspec = json.loads(f.read())
    self.decoders = decoders
    self.parsedspec = parse_spec(self.rawspec)
    if cache_refs:
      fp = io.BytesIO((directory / 'refs.bag').read_bytes())
      self.refreader = BagReader(fp, cache_index)
    else:
      self.refreader = BagReader(directory / 'refs.bag', cache_index)
    self.readers = {
        k: BagReader(directory / f'{k}.bag', cache_index)
        for k in self.parsedspec.keys()}

  @property
  def spec(self):
    return self.rawspec

  @property
  def size(self):
    return sum(x.size for x in self.readers.values()) + self.refreader.size

  def __len__(self):
    return len(self.refreader)

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index, mask = index
      assert isinstance(mask, dict), mask
    else:
      mask = {k: True for k in self.parsedspec.keys()}
    assert all(k in self.parsedspec for k in mask), (self.parsedspec, mask)
    refs = self.refreader[index]
    refs = msgpack.unpackb(refs)
    datapoint = {}
    for i, (key, (islist, dtype, args)) in enumerate(self.parsedspec.items()):
      msk = mask.get(key, False)
      reader = self.readers[key]
      if islist:
        if not msk:
          continue
        # Construct available range.
        if refs[i]:
          start, length = refs[i]
          avail = range(start, start + length)
        else:
          avail = range(0)
        # Select requested range.
        if isinstance(msk, bool):
          requested = avail
        elif isinstance(msk, (slice, range)):
          assert msk.start >= 0 and msk.stop >= 0 and msk.step == 1, msk
          requested = range(
              avail.start + msk.start,
              min(avail.start + msk.stop, avail.stop))
        else:
          raise TypeError((msk, type(msk)))
        # Request range if needed.
        if requested:
          assert isinstance(requested, range)
          values = reader[requested]
          if self.decoders is not None:
            values = [self.decoders[dtype](x, *args) for x in values]
          datapoint[key] = values
        else:
          datapoint[key] = []
      else:
        if not isinstance(msk, bool):
          raise TypeError((key, mask))
        if not msk:
          continue
        idx = refs[i]
        assert isinstance(idx, int)
        value = reader[idx]
        if self.decoders is not None:
          value = self.decoders[dtype](value, *args)
        datapoint[key] = value
    return datapoint

  def close(self):
    self.refreader.close()
    for reader in self.readers.values():
      reader.close()


class BagWriter(Closing):

  def __init__(self, fp):
    super().__init__()
    if isinstance(fp, str):
      fp = pathlib.Path(fp)
    if hasattr(fp, '__fspath__'):
      assert not fp.exists(), fp
      fp = fp.open('wb')
    assert isinstance(fp, io.BufferedIOBase), type(fp)
    assert fp.writable(), fp
    self.fp = fp
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
      self.fp.write(self.towrite[0])
    else:
      self.fp.write(b''.join(self.towrite))
    self.towrite.clear()

  def close(self):
    assert not self.closed
    self.limits.append(self.offset)
    self.towrite.extend(tuple(x.to_bytes(8, 'little') for x in self.limits))
    self.flush()
    self.fp.close()


class BagReader(Closing):

  def __init__(self, fp, cache_index=True):
    super().__init__()
    if isinstance(fp, str):
      fp = pathlib.Path(fp)
    if hasattr(fp, '__fspath__'):
      fp = fp.open('rb')
    assert isinstance(fp, io.BufferedIOBase), type(fp)
    assert fp.readable(), fp
    fp.seek(-8, os.SEEK_END)
    self.filesize = fp.tell() + 8
    self.recordend = int.from_bytes(fp.read(8), 'little')
    self.length = (self.filesize - 8 - self.recordend) // 8
    if cache_index:
      fp.seek(self.recordend, os.SEEK_SET)
      self.limits = fp.read(8 * self.length + 8)
    else:
      self.limits = None
    self.fp = fp

  @property
  def size(self):
    return self.filesize

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert not self.closed
    assert isinstance(index, (int, slice, range)), index
    if isinstance(index, int):
      start = self._get_start(index)
      if index + 1 < self.length:
        end = self._get_start(index + 1)
      else:
        end = self.recordend
      self.fp.seek(start, os.SEEK_SET)
      return self.fp.read(end - start)
    else:
      assert index.start >= 0 and index.stop >= 0 and index.step == 1, index
      index = range(index.start, min(index.stop, self.length))
      if index.start == index.stop:
        return []
      limits = self._get_limits(index.start, index.stop + 1)
      start, stop = limits[0], limits[-1]
      self.fp.seek(start, os.SEEK_SET)
      buffer = self.fp.read(stop - start)
      records = [
          buffer[i - start: j - start]
          for i, j in zip(limits[:-1], limits[1:])]
      return records

  def close(self):
    assert not self.closed
    self.fp.close()

  def _get_start(self, index):
    if self.limits:
      return int.from_bytes(self.limits[8 * index: 8 * (index + 1)], 'little')
    else:
      self.fp.seek(-8 * (self.length - index) - 8, os.SEEK_END)
      offset = self.fp.read(8)
      offset = int.from_bytes(offset, 'little')
      return offset

  def _get_limits(self, start, stop):
    if self.limits:
      return [
          int.from_bytes(self.limits[8 * i: 8 * (i + 1)], 'little')
          for i in range(start, stop)]
    else:
      self.fp.seek(-8 * (self.length - start) - 8, os.SEEK_END)
      limits = self.fp.read(8 * (stop - start))
      limits = [
          int.from_bytes(limits[8 * i: 8 * (i + 1)], 'little')
          for i in range(stop - start)]
      return limits
