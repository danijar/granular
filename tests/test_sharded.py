import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import granular


class TestSharded:

  @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
  def test_writer(self, tmpdir, shardlen):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    with granular.ShardedDatasetWriter(
        directory, spec, granular.encoders, shardlen) as writer:
      assert writer.spec == spec
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
      if shardlen:
        assert writer.shards == int(np.ceil(10 / shardlen))
      else:
        assert writer.shards == 1
    assert set(x.name for x in directory.glob('*')) == {
        f'{i:06}' for i in range(writer.shards)}
    for folder in directory.glob('*'):
      assert set(x.name for x in folder.glob('*')) == {
          'spec.json', 'refs.bag', 'bar.bag', 'baz.bag', 'foo.bag'}

  @pytest.mark.parametrize('shardlen', (None, 1, 5))
  def test_length(self, tmpdir, shardlen):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    with granular.ShardedDatasetWriter(
        directory, spec, granular.encoders, shardlen) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
      if shardlen:
        assert writer.shards == int(np.ceil(10 / shardlen))
      else:
        assert writer.shards == 1

  @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
  @pytest.mark.parametrize('parallel', (True, False))
  def test_roundtrip(self, tmpdir, shardlen, parallel):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.ShardedDatasetWriter(
        directory, spec, granular.encoders, shardlen) as writer:
      assert writer.spec == spec
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      shards = writer.shards
      size = writer.size
    with granular.ShardedDatasetReader(
        directory, granular.decoders, parallel=parallel) as reader:
      assert reader.spec == spec
      assert reader.shards == shards
      assert reader.size == size
      assert len(reader) == 10
      for i in range(10):
        assert reader[i] == datapoints[i]

  @pytest.mark.parametrize('shardlen', (1, 3, 10, 15))
  @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
  def test_worker_writing(self, tmpdir, shardlen, nworkers):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]'}
    datapoints = [{'bar': i, 'baz': ['hello'] * i} for i in range(10)]
    shards = 0
    size = 0
    for worker in range(nworkers):
      with granular.ShardedDatasetWriter(
          directory, spec, granular.encoders, shardlen,
          shardstart=worker, shardstep=nworkers) as writer:
        for i in range(worker, 10, nworkers):
          writer.append(datapoints[i])
        shards += writer.shards
        size += writer.size
    with granular.ShardedDatasetReader(directory, granular.decoders) as reader:
      assert reader.shards == shards
      assert reader.size == size
      assert len(reader) == 10
      received = [reader[i] for i in range(10)]
      received = sorted(received, key=lambda x: x['bar'])
      assert datapoints == received

  @pytest.mark.parametrize('shardlen', (1, 3, 10, 15))
  @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
  def test_worker_roundtrip(self, tmpdir, shardlen, nworkers):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]'}
    datapoints = [{'bar': i, 'baz': ['hello'] * i} for i in range(10)]
    shards = 0
    size = 0
    for worker in range(nworkers):
      with granular.ShardedDatasetWriter(
          directory, spec, granular.encoders, shardlen,
          shardstart=worker, shardstep=nworkers) as writer:
        for i in range(worker, 10, nworkers):
          writer.append(datapoints[i])
        shards += writer.shards
        size += writer.size
    received = []
    for worker in range(nworkers):
      with granular.ShardedDatasetReader(
          directory, granular.decoders,
          shardstart=worker, shardstep=nworkers) as reader:
        received += [reader[i] for i in range(len(reader))]
    received = sorted(received, key=lambda x: x['bar'])
    assert datapoints == received

  @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
  def test_masks(self, tmpdir, shardlen):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.ShardedDatasetWriter(
        directory, spec, granular.encoders, shardlen) as writer:
      assert writer.spec == spec
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
    with granular.ShardedDatasetReader(directory, granular.decoders) as reader:
      for i in range(10):
        mask = reader.mask(i)
        assert mask == {'bar': True, 'baz': range(i), 'foo': True}
        datapoint = reader[i, mask]
        assert datapoint == datapoints[i]
