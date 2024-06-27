import io
import itertools
import json
import operator
import os
import pathlib
import re

import msgpack


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


class DatasetWriter:

  def __init__(self, directory, spec, encoders, shardsize):
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    assert not directory.exists(), directory
    directory.mkdir(parents=True)
    spec = dict(sorted(spec.items(), key=lambda x: x[0]))
    self.directory = directory
    self.encoders = encoders
    self.rawspec = spec
    self.spec = parse_spec(spec)
    self.refwriter = ShardedWriter(
        self.directory / 'refs.bag', shardsize)
    self.writers = {
        k: ShardedWriter(self.directory / f'{k}.bag', shardsize)
        for k in self.spec.keys()}
    self.specwritten = False
    self.closed = False

  @property
  def size(self):
    return sum(x.size for x in self.writers.values()) + self.refwriter.size

  def __len__(self):
    return len(self.refwriter)

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

  def append(self, datapoint, flush=True):
    assert isinstance(datapoint, dict)
    assert set(datapoint.keys()) == set(self.spec.keys()), (
        datapoint.keys(), self.spec)
    refs = []
    # Iterate in sorted key order.
    for key, (islist, dtype, args) in self.spec.items():
      writer = self.writers[key]
      if islist:
        # Hold only one value of the generator in memory at a time.
        indices = []
        for value in datapoint[key]:
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
      content = json.dumps(self.rawspec).encode('utf-8')
      (self.directory / 'spec.json').write_bytes(content)
    self.refwriter.flush()
    for writer in self.writers.values():
      writer.flush()

  def close(self):
    self.closed = True
    self.refwriter.close()
    for writer in self.writers.values():
      writer.close()


class DatasetReader:

  def __init__(self, directory, decoders, cache_index=True):
    if isinstance(directory, str):
      directory = pathlib.Path(directory)
    content = (directory / 'spec.json').read_bytes()
    spec = json.loads(content)
    self.decoders = decoders
    self.spec = parse_spec(spec)
    self.refreader = ShardedReader(directory / 'refs.bag', cache_index)
    self.readers = {
        k: ShardedReader(directory / f'{k}.bag', cache_index)
        for k in self.spec.keys()}
    self.closed = False

  @property
  def size(self):
    return sum(x.size for x in self.readers.values()) + self.refreader.size

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

  def __len__(self):
    return len(self.refreader)

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index, mask = index
      assert isinstance(mask, dict), mask
    else:
      mask = {k: True for k in self.spec.keys()}
    assert all(k in self.spec for k in mask), (self.spec, mask)
    refs = self.refreader[index]
    refs = msgpack.unpackb(refs)
    datapoint = {}
    for i, (key, (islist, dtype, args)) in enumerate(self.spec.items()):
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
        value = self.decoders[dtype](value, *args)
        datapoint[key] = value
    return datapoint

  def close(self):
    assert not self.closed
    self.closed = True
    self.refreader.close()
    for reader in self.readers.values():
      reader.close()


class ShardedWriter:

  def __init__(self, filename, shardsize):
    if isinstance(filename, str):
      filename = pathlib.Path(filename)
    self.directory = filename.parent
    self.stem = filename.stem
    self.suffix = filename.suffix
    self.shardsize = shardsize
    self.shardnum = 0
    self.writer = None
    self.closed = False
    self.prevsize = 0
    self.previndex = 0

  @property
  def size(self):
    return self.prevsize + self.writer.size

  @property
  def shards(self):
    return self.shardnum + 1

  def __len__(self):
    return self.previndex + len(self.writer)

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

  def append(self, record, flush=True):
    newsize = self.writer and self.writer.size + len(record) + 8 + 8
    if self.writer and newsize > self.shardsize:
      assert len(self.writer) > 0, (newsize, self.shardsize)
      self.prevsize += self.writer.size
      self.previndex += len(self.writer)
      self.writer.close()
      self.writer = None
      self.shardnum += 1
    if not self.writer:
      assert self.shardnum < int(1e6), self.shardnum
      name = f'{self.stem}-{self.shardnum:05}{self.suffix}'
      fp = (self.directory / name).open('wb')
      self.writer = Writer(fp)
    index = self.writer.append(record, flush)
    return self.previndex + index

  def flush(self):
    if self.writer:
      self.writer.flush()

  def close(self):
    self.closed = True
    if self.writer:
      self.writer.close()


class ShardedReader:

  def __init__(self, filename, cache_index=True):
    if isinstance(filename, str):
      filename = pathlib.Path(filename)
    pattern = str(filename.stem + '-*' + filename.suffix)
    filenames = sorted(filename.parent.glob(pattern))
    self.readers = [Reader(x, cache_index) for x in filenames]
    self.idxoffsets = [0] + list(itertools.accumulate(
        [len(x) for x in self.readers[:-1]], operator.add))
    self.closed = False

  @property
  def size(self):
    return sum(x.size for x in self.readers)

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

  def __len__(self):
    return self.idxoffsets[-1] + len(self.readers[-1])

  def __getitem__(self, index):
    assert not self.closed
    if isinstance(index, int):
      for offset, reader in zip(self.idxoffsets, self.readers):
        if index < offset + len(reader):
          break
      return reader[index - offset]
    else:
      assert index.start >= 0 and index.stop >= 0 and index.step == 1, index
      requests = []
      i = index
      for offset, reader in zip(self.idxoffsets, self.readers):
        if i.stop <= offset + len(reader):
          requests.append((reader, range(i.start - offset, i.stop - offset)))
          break
        elif i.start < offset + len(reader):
          requests.append((reader, range(i.start - offset, len(reader))))
          i = range(offset + len(reader), i.stop)
      records = []
      for reader, i in requests:
        records.extend(reader[i])
      return records

  def close(self):
    assert not self.closed
    self.closed = True
    for reader in self.readers:
      reader.close()


class Writer:

  def __init__(self, fp):
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
    self.closed = False
    self.towrite = []

  @property
  def size(self):
    return self.offset + 8 * self.length + 8

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

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
    self.closed = True
    self.limits.append(self.offset)
    self.towrite.extend(tuple(x.to_bytes(8, 'little') for x in self.limits))
    self.flush()
    self.fp.close()


class Reader:

  def __init__(self, fp, cache_index=True):
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
    self.closed = False

  @property
  def size(self):
    return self.filesize

  def __enter__(self):
    assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()

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
    self.closed = True
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
