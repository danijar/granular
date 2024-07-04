import concurrent.futures
import itertools
import json
import operator
import pathlib
import pickle

from . import utils
from . import dataset


class ShardedDatasetWriter(utils.Closing):

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
      self.writer = dataset.DatasetWriter(folder, self.spec, self.encoders)
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


class ShardedDatasetReader(utils.Closing):

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
      make = lambda x: dataset.DatasetReader(
          x, decoders, cache_index, cache_keys, parallel)
      with concurrent.futures.ThreadPoolExecutor(len(selected)) as pool:
        self.readers = list(pool.map(make, selected))
    else:
      self.readers = [
          dataset.DatasetReader(x, decoders, cache_index, cache_keys, parallel)
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

  def available(self, index):
    reader, local_index = self._resolve(index)
    return reader.available(local_index)

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
