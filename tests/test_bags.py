import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import bags


class TestBags:

  def test_single_writer(self, tmpdir):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    total = 8
    with bags.BagWriter(filename) as writer:
      for i in range(100):
        size = rng.integers(4, 100)
        value = i.to_bytes(size, 'little')
        index = writer.append(value)
        assert index == i
        assert len(writer) == i + 1
        total += size + 8
      assert writer.size == total
    assert filename.exists()
    assert filename.stat().st_size == total
    with bags.BagReader(filename) as reader:
      reader.size == total

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_single_roundtrip(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    values = []
    total = 0
    with bags.BagWriter(filename) as writer:
      for i in range(100):
        size = int(rng.integers(4, 100))
        value = int(rng.integers(0, 1000))
        writer.append(value.to_bytes(size, 'little'))
        values.append(value)
        total += size
    with bags.BagReader(filename, cache_index) as reader:
      assert len(reader) == 100
      for index, reference in enumerate(values):
        value = reader[index]
        value = int.from_bytes(value, 'little')
        assert value == reference

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_single_reader_slicing(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    with bags.BagWriter(filename) as writer:
      for i in range(100):
        writer.append(i.to_bytes(int(rng.integers(4, 32)), 'little'))
    with bags.BagReader(filename, cache_index) as reader:
      assert len(reader) == 100
      for requested in (
          range(0),
          range(0, 1),
          range(0, 10),
          range(3, 5),
          range(90, 100),
          range(90, 110),
      ):
        values = reader[requested]
        values = [int.from_bytes(x, 'little') for x in values]
        expected = [x for x in list(requested) if 0 <= x < 100]
        assert values == expected

  # def test_sharded_writer(
  #     self, tmpdir, shardsize=100, itemsize=20, numitems=10):
  #   directory = pathlib.Path(tmpdir)
  #   filename = directory / 'file.bag'
  #   with bags.ShardedWriter(filename, shardsize) as writer:
  #     for i in range(numitems):
  #       index = writer.append(i.to_bytes(itemsize, 'little'))
  #       assert index == i
  #       assert len(writer) == i + 1
  #   shardlen = (shardsize - 8) // (itemsize + 8)
  #   shardsize = shardlen * (itemsize + 8) + 8
  #   numshards = int(np.ceil(numitems / shardlen))
  #   filenames = sorted(directory.glob('*'))
  #   assert len(filenames) == numshards
  #   for i, filename in enumerate(filenames):
  #     assert filename == directory / f'file-{i:>05}.bag'
  #   for filename in filenames[:-1]:
  #     assert filename.stat().st_size == shardsize
  #   assert filenames[-1].stat().st_size <= shardsize

  # @pytest.mark.parametrize('cache_index', (True, False))
  # def test_sharded_roundtrip(
  #     self, tmpdir, cache_index, shardsize=100, itemsize=20, numitems=10):
  #   directory = pathlib.Path(tmpdir)
  #   filename = directory / 'file.bag'
  #   with bags.ShardedWriter(filename, shardsize) as writer:
  #     for i in range(numitems):
  #       writer.append(i.to_bytes(itemsize, 'little'))
  #   shardlen = (shardsize - 8) // (itemsize + 8)
  #   numshards = int(np.ceil(numitems / shardlen))
  #   with bags.ShardedReader(filename, cache_index) as reader:
  #     assert len(reader) == numitems
  #     assert reader.size == (itemsize + 8) * numitems + 8 * numshards
  #     for i in range(numitems):
  #       assert int.from_bytes(reader[i], 'little') == i

  # @pytest.mark.parametrize('cache_index', (True, False))
  # @pytest.mark.parametrize('shardsize', (128, 512))
  # def test_sharded_reader_slicing(self, tmpdir, cache_index, shardsize):
  #   filename = pathlib.Path(tmpdir) / 'file.bag'
  #   rng = np.random.default_rng(seed=0)
  #   with bags.ShardedWriter(filename, shardsize) as writer:
  #     for i in range(100):
  #       writer.append(i.to_bytes(int(rng.integers(4, 32)), 'little'))
  #     assert writer.shards > 1
  #   with bags.ShardedReader(filename, cache_index) as reader:
  #     assert len(reader) == 100
  #     for requested in (
  #         range(0),
  #         range(0, 1),
  #         range(0, 10),
  #         range(3, 5),
  #         range(90, 100),
  #         range(90, 110),
  #     ):
  #       values = reader[requested]
  #       values = [int.from_bytes(x, 'little') for x in values]
  #       expected = [x for x in list(requested) if 0 <= x < 100]
  #       assert values == expected

  def test_dataset_writer(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int(4)', 'baz': 'utf8[]'}
    with bags.DatasetWriter(directory, spec, bags.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        index = writer.append({'foo': 'hello world', 'bar': i, 'baz': baz})
        assert index == i
        assert len(writer) == i + 1
      assert len(writer) == 10
      assert writer.size > 0
    assert set(x.name for x in directory.glob('*')) == {
        'spec.json', 'refs.bag', 'foo.bag', 'bar.bag', 'baz.bag'}
    spec2 = json.loads((directory / 'spec.json').read_bytes())
    assert list(spec2.keys()) == sorted(spec2.keys())
    assert set(spec.keys()) == set(spec2.keys())
    assert spec == spec2

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_dataset_roundtrip(self, tmpdir, cache_index):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with bags.DatasetWriter(directory, spec, bags.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      size = writer.size
    with bags.DatasetReader(directory, bags.decoders, cache_index) as reader:
      assert len(reader) == 10
      assert reader.size == size
      for i in range(10):
        datapoint = reader[i]
        assert datapoint == datapoints[i]

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_dataset_slicing(self, tmpdir, cache_index):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int(4)', 'baz': 'utf8[]'}
    with bags.DatasetWriter(directory, spec, bags.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
    with bags.DatasetReader(directory, bags.decoders, cache_index) as reader:
      assert reader[3, {}] == {}
      assert reader[3, {'foo': True}] == {'foo': 'hello world'}
      with pytest.raises(TypeError):
        assert reader[3, {'foo': 12}]
      assert reader[3, {'foo': True, 'baz': True}] == {
          'baz': ['word0', 'word1', 'word2'],
          'foo': 'hello world'}
      assert reader[3, {'baz': range(1)}] == {'baz': ['word0']}
      assert reader[3, {'baz': range(1, 10)}] == {'baz': ['word1', 'word2']}
      with pytest.raises(TypeError):
        assert reader[3, {'bar': range(1)}]

  @pytest.mark.parametrize('shard_size', (None, 1, 200, 1000))
  def test_sharded_writer(self, tmpdir, shard_size):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]', 'foo': 'utf8'}
    with bags.ShardedDatasetWriter(
        directory, spec, bags.encoders, shard_size) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
      shards = writer.shards
    if shard_size == 1:
      assert shards == 10
    elif shard_size is None or shard_size == 1000:
      assert shards == 1
    else:
      assert 2 <= shards <= 8
    assert set(x.name for x in directory.glob('*')) == {
        f'{i:06}' for i in range(writer.shards)}
    for folder in directory.glob('*'):
      assert set(x.name for x in folder.glob('*')) == {
          'spec.json', 'refs.bag', 'bar.bag', 'baz.bag', 'foo.bag'}

  @pytest.mark.parametrize('shard_length', (None, 1, 5))
  def test_sharded_writer_length(self, tmpdir, shard_length):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]', 'foo': 'utf8'}
    with bags.ShardedDatasetWriter(
        directory, spec, bags.encoders,
        shard_length=shard_length) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
      if shard_length:
        assert writer.shards == int(np.ceil(10 / shard_length))
      else:
        assert writer.shards == 1

  @pytest.mark.parametrize('shard_size', (None, 1, 200, 1000))
  def test_sharded_roundtrip(self, tmpdir, shard_size):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with bags.ShardedDatasetWriter(
        directory, spec, bags.encoders, shard_size) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      shards = writer.shards
      size = writer.size
    with bags.ShardedDatasetReader(directory, bags.decoders) as reader:
      assert reader.shards == shards
      assert reader.size == size
      assert len(reader) == 10
      for i in range(10):
        assert reader[i] == datapoints[i]

  @pytest.mark.parametrize('shard_size', (None, 1, 200, 1000))
  @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
  def test_distributed_writer(self, tmpdir, shard_size, nworkers):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]'}
    datapoints = [{'bar': i, 'baz': ['hello'] * i} for i in range(10)]
    shards = 0
    size = 0
    for worker in range(nworkers):
      with bags.ShardedDatasetWriter(
          directory, spec, bags.encoders, shard_size,
          shard_start=worker, shard_step=nworkers) as writer:
        for i in range(worker, 10, nworkers):
          writer.append(datapoints[i])
        shards += writer.shards
        size += writer.size
    with bags.ShardedDatasetReader(directory, bags.decoders) as reader:
      assert reader.shards == shards
      assert reader.size == size
      assert len(reader) == 10
      received = [reader[i] for i in range(10)]
      received = sorted(received, key=lambda x: x['bar'])
      assert datapoints == received

  @pytest.mark.parametrize('shard_size', (None, 1, 200, 1000))
  @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
  def test_distributed_roundtrip(self, tmpdir, shard_size, nworkers):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]'}
    datapoints = [{'bar': i, 'baz': ['hello'] * i} for i in range(10)]
    shards = 0
    size = 0
    for worker in range(nworkers):
      with bags.ShardedDatasetWriter(
          directory, spec, bags.encoders, shard_size,
          shard_start=worker, shard_step=nworkers) as writer:
        for i in range(worker, 10, nworkers):
          writer.append(datapoints[i])
        shards += writer.shards
        size += writer.size
    received = []
    for worker in range(nworkers):
      with bags.ShardedDatasetReader(
          directory, bags.decoders,
          shard_start=worker, shard_step=nworkers) as reader:
        received += [reader[i] for i in range(len(reader))]
    received = sorted(received, key=lambda x: x['bar'])
    assert datapoints == received

  def test_encoders_decoders(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {
        'a': 'utf8',
        'b': 'int(4)',
        'c': 'utf8[]',
        'd': 'msgpack',
        'e': 'int[]',
        'f': 'bytes',
        'g': 'array(float32,10,4)',
        'h': 'jpeg(80)',
        'i': 'png',
        'j': 'mp4(15)',
    }
    datapoints = []
    for i in range(10):
      datapoints.append({
          'a': 'hello world',
          'b': i,
          'c': [f'word{j}' for j in range(i)],
          'd': {'foo': 'bar', 'baz': 12},
          'e': list(range(i)),
          'f': b'hello world',
          'g': np.ones((10, 4), np.float32),
          'h': np.zeros((320, 180, 3), np.uint8),
          'i': np.zeros((80, 60, 4), np.uint8),
          'j': np.zeros((20, 80, 60, 3), np.uint8),
      })
    with bags.DatasetWriter(directory, spec, bags.encoders) as writer:
      for datapoint in datapoints:
        writer.append(datapoint)
    with bags.DatasetReader(directory, bags.decoders) as reader:
      for i in range(len(reader)):
        actual = reader[i]
        reference = datapoints[i]
        assert actual.keys() == reference.keys()
        for key in actual.keys():
          if isinstance(actual[key], np.ndarray):
            assert (actual[key] == reference[key]).all()
          else:
            assert actual[key] == reference[key]
