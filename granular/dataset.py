import concurrent.futures
import functools
import json
import pathlib
import pickle
import re

import msgpack

from . import utils
from . import bag


class DatasetWriter(utils.Closing):

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
    self.refwriter = bag.BagWriter(self.directory / 'refs.bag')
    self.writers = {
        k: bag.BagWriter(self.directory / f'{k}.bag')
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


class DatasetReader(utils.Closing):

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
        k: bag.BagReader(
            directory / f'{k}.bag', cache_index, cache_data=(k in cache_keys))
        for k in ('refs', *self.spec.keys())}
    if decoders is None:
      decoders = {k: None for k in self.spec.keys()}
    else:
      decoders = {k: decoders[v.rstrip('[]')] for k, v in self.spec.items()}
    self.decoders = decoders
    self.parallel = parallel
    if parallel:
      self.pool = concurrent.futures.ThreadPoolExecutor(len(self.spec))

  @property
  def spec(self):
    return self.thespec

  @property
  def size(self):
    return sum(x.size for x in self.readers.values())

  def available(self, index):
    return {
        key: range(0, ref[1] if ref else 0) if dtype.endswith('[]') else True
        for ref, (key, dtype) in zip(self._getref(index), self.spec.items())}

  def __getstate__(self):
    d = self.__dict__.copy()
    if self.parallel:
      d.pop('pool')
    return d

  def __setstate__(self, d):
    self.__dict__.update(d)
    if self.parallel:
      self.pool = concurrent.futures.ThreadPoolExecutor(len(self.spec))

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
    if self.parallel:
      self.pool.shutdown(wait=False)
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
