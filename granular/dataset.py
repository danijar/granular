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
    self.cache_keys = cache_keys
    self.workers = int(parallel and len(set(self.spec) - set(cache_keys)))
    if self.workers:
      self.pool = concurrent.futures.ThreadPoolExecutor(self.workers)

  @property
  def spec(self):
    return self.thespec

  @property
  def size(self):
    return sum(x.size for x in self.readers.values())

  def available(self, index):
    ref = self._getrefs(index, index + 1)[0]
    return {
        key: range(0, ref[1] if ref else 0) if dtype.endswith('[]') else True
        for ref, (key, dtype) in zip(ref, self.spec.items())}

  def __getstate__(self):
    d = self.__dict__.copy()
    if self.workers:
      d.pop('pool')
    return d

  def __setstate__(self, d):
    self.__dict__.update(d)
    if self.workers:
      self.pool = concurrent.futures.ThreadPoolExecutor(self.workers)

  def __len__(self):
    return len(self.readers['refs'])

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index, mask = index
      assert isinstance(mask, dict), mask
    else:
      mask = {k: True for k in self.spec}
    assert all(k in self.spec for k in mask), (self.spec, mask)
    needed = {}
    if isinstance(index, int):
      refs = self._getrefs(index, index + 1)[0]
      for i, (key, dtype) in enumerate(self.spec.items()):
        ref, msk = refs[i], mask.get(key, False)
        if msk is False:
          continue
        elif dtype.endswith('[]'):
          assert isinstance(msk, (bool, slice, range)), (key, msk, type(msk))
          begin, end = (ref[0], ref[0] + ref[1]) if ref else (0, 0)
          if msk is True:
            needed[key] = range(begin, end)
          else:
            assert 0 <= msk.start <= msk.stop and msk.step in (None, 1), msk
            needed[key] = range(begin + msk.start, min(begin + msk.stop, end))
        else:
          assert isinstance(msk, bool) and msk is True, (key, msk, type(msk))
          needed[key] = ref
      point = self._fetch(needed)
      decoded = {k: self._decode(k, v, self.spec[k]) for k, v in point.items()}
    else:
      assert 0 <= index.start <= index.stop and index.step in (None, 1), index
      refs = self._getrefs(index.start, index.stop)
      for i, (key, dtype) in enumerate(self.spec.items()):
        # Cannot range read datapoints that contain sequence modalities,
        # because they may not be consecutive and thus could be slow.
        assert not dtype.endswith('[]'), (index, key, dtype)
        ref, msk = [x[i] for x in refs], mask.get(key, False)
        assert isinstance(msk, bool), (key, msk, type(msk))
        if msk:
          needed[key] = range(ref[0], ref[-1] + 1)
      points = self._fetch(needed)
      decoded = {
          k: [self._decode(k, v, self.spec[k]) for v in vs]
          for k, vs in points.items()}
    return decoded

  def copy(self):
    return pickle.loads(pickle.dumps(self))

  def close(self):
    if self.workers:
      self.pool.shutdown(wait=False)
    for reader in self.readers.values():
      reader.close()

  @functools.lru_cache(maxsize=1)
  def _getrefs(self, start, stop):
    assert 0 <= start <= stop <= len(self), (start, stop, len(self))
    refs = self.readers['refs'][start: stop]
    refs = [msgpack.unpackb(x) for x in refs]
    assert len(refs) == stop - start
    for ref in refs:
      assert len(ref) == len(self.spec)
      for (key, dtype), r in zip(self.spec.items(), ref):
        if dtype.endswith('[]'):
          assert isinstance(r, list) and len(r) in (0, 2)
        else:
          assert isinstance(r, int)
    return refs

  def _needed(self, key, dtype, ref, mask):
    assert isinstance(mask, (bool, slice, range)), (mask, type(mask))
    assert mask is not False, mask
    if dtype.endswith('[]'):
      begin, end = (ref[0], ref[0] + ref[1]) if ref else (0, 0)
      if mask is True:
        return range(begin, end)
      else:
        assert 0 <= mask.start <= mask.stop and mask.step in (1, None), mask
        return range(begin + mask.start, min(begin + mask.stop, end))
    else:
      return ref

  def _fetch(self, needed):
    if self.workers:
      reqs1 = {k: v for k, v in needed.items() if k not in self.cache_keys}
      reqs2 = {k: v for k, v in needed.items() if k in self.cache_keys}
      values1 = self.pool.map(
          lambda it: self.readers[it[0]][it[1]], reqs1.items())
      values2 = [self.readers[k][r] for k, r in reqs2.items()]
      datapoint = dict(
          list(zip(reqs1.keys(), values1)) +
          list(zip(reqs2.keys(), values2)))
    else:
      datapoint = {k: self.readers[k][r] for k, r in needed.items()}
    return datapoint

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
