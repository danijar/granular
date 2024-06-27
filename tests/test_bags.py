import json
import pathlib
import sys

import msgpack
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import bags


ENCODERS = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'int': lambda x, size=None: x.to_bytes(
        int(size if size else np.ceil(np.log2(1 + x) / 8))),
    'msgpack': msgpack.packb,
    'array': lambda x, *args: x.tobytes(),
}

DECODERS = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'int': lambda x, size=None: int.from_bytes(x),
    'msgpack': msgpack.unpackb,
    'array': lambda x, dtype, *shape: np.frombuffer(x, dtype).reshape(
        tuple(int(x) for x in shape))
}


class TestBags:

  def test_single_writer(self, tmpdir):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    total = 8
    with bags.Writer(filename) as writer:
      for i in range(100):
        size = rng.integers(4, 100)
        value = i.to_bytes(size)
        index = writer.append(value)
        assert index == i
        assert len(writer) == i + 1
        total += size + 8
      assert writer.size == total
    assert filename.exists()
    assert filename.stat().st_size == total
    with bags.Reader(filename) as reader:
      reader.size == total

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_single_roundtrip(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    values = []
    total = 0
    with bags.Writer(filename) as writer:
      for i in range(100):
        size = int(rng.integers(4, 100))
        value = int(rng.integers(0, 1000))
        writer.append(value.to_bytes(size))
        values.append(value)
        total += size
    with bags.Reader(filename, cache_index) as reader:
      assert len(reader) == 100
      for index, reference in enumerate(values):
        value = reader[index]
        value = int.from_bytes(value)
        assert value == reference

  def test_sharded_writer(
      self, tmpdir, shardsize=100, itemsize=20, numitems=10):
    directory = pathlib.Path(tmpdir)
    filename = directory / 'file.bag'
    with bags.ShardedWriter(filename, shardsize) as writer:
      for i in range(numitems):
        index = writer.append(i.to_bytes(itemsize))
        assert index == i
        assert len(writer) == i + 1
    shardlen = (shardsize - 8) // (itemsize + 8)
    shardsize = shardlen * (itemsize + 8) + 8
    numshards = int(np.ceil(numitems / shardlen))
    filenames = sorted(directory.glob('*'))
    assert len(filenames) == numshards
    for i, filename in enumerate(filenames):
      assert filename == directory / f'file-{i:>05}.bag'
    for filename in filenames[:-1]:
      assert filename.stat().st_size == shardsize
    assert filenames[-1].stat().st_size <= shardsize

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_sharded_roundtrip(
      self, tmpdir, cache_index, shardsize=100, itemsize=20, numitems=10):
    directory = pathlib.Path(tmpdir)
    filename = directory / 'file.bag'
    with bags.ShardedWriter(filename, shardsize) as writer:
      for i in range(numitems):
        writer.append(i.to_bytes(itemsize))
    shardlen = (shardsize - 8) // (itemsize + 8)
    numshards = int(np.ceil(numitems / shardlen))
    with bags.ShardedReader(filename, cache_index) as reader:
      assert len(reader) == numitems
      assert reader.size == (itemsize + 8) * numitems + 8 * numshards
      for i in range(numitems):
        assert int.from_bytes(reader[i]) == i

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_single_reader_slicing(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    with bags.Writer(filename) as writer:
      for i in range(100):
        writer.append(i.to_bytes(int(rng.integers(4, 32))))
    with bags.Reader(filename, cache_index) as reader:
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
        values = [int.from_bytes(x) for x in values]
        expected = [x for x in list(requested) if 0 <= x < 100]
        assert values == expected

  @pytest.mark.parametrize('cache_index', (True, False))
  @pytest.mark.parametrize('shardsize', (128, 512))
  def test_sharded_reader_slicing(self, tmpdir, cache_index, shardsize):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    with bags.ShardedWriter(filename, shardsize) as writer:
      for i in range(100):
        writer.append(i.to_bytes(int(rng.integers(4, 32))))
      assert writer.shards > 1
    with bags.ShardedReader(filename, cache_index) as reader:
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
        values = [int.from_bytes(x) for x in values]
        expected = [x for x in list(requested) if 0 <= x < 100]
        assert values == expected

  def test_dataset_writer(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int(4)', 'baz': 'utf8[]'}
    with bags.DatasetWriter(
        directory, spec, ENCODERS, shardsize=100) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        index = writer.append({'foo': 'hello world', 'bar': i, 'baz': baz})
        assert index == i
        assert len(writer) == i + 1
      assert len(writer) == 10
      assert writer.size > 0
    spec2 = json.loads((directory / 'spec.json').read_bytes())
    assert list(spec2.keys()) == sorted(spec2.keys())
    assert set(spec.keys()) == set(spec2.keys())
    assert spec == spec2
    assert len(list(directory.glob('refs-*.bag'))) == 2
    assert len(list(directory.glob('bar-*.bag'))) == 2
    assert len(list(directory.glob('baz-*.bag'))) == 8
    assert len(list(directory.glob('foo-*.bag'))) == 3
    assert len(list(directory.glob('*'))) == 1 + 2 + 2 + 8 + 3

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_dataset_roundtrip(self, tmpdir, cache_index):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int(4)', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with bags.DatasetWriter(
        directory, spec, ENCODERS, shardsize=100) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      size = writer.size
    with bags.DatasetReader(directory, DECODERS, cache_index) as reader:
      assert len(reader) == 10
      assert reader.size == size
      for i in range(10):
        datapoint = reader[i]
        assert datapoint == datapoints[i]

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_dataset_slicing(self, tmpdir, cache_index):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int(4)', 'baz': 'utf8[]'}
    datapoints = []
    with bags.DatasetWriter(
        directory, spec, ENCODERS, shardsize=100) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
    with bags.DatasetReader(directory, DECODERS, cache_index) as reader:
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
      })
    with bags.DatasetWriter(
        directory, spec, ENCODERS, shardsize=1000) as writer:
      for datapoint in datapoints:
        writer.append(datapoint)
    with bags.DatasetReader(directory, DECODERS) as reader:
      for i in range(len(reader)):
        actual = reader[i]
        reference = datapoints[i]
        assert actual.keys() == reference.keys()
        for key in actual.keys():
          if isinstance(actual[key], np.ndarray):
            assert (actual[key] == reference[key]).all()
          else:
            assert actual[key] == reference[key]
