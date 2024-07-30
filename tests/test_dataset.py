import json
import pathlib

import cloudpickle
import granular
import pytest


class TestDataset:

  def test_writer(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int', 'baz': 'utf8[]'}
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
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
  @pytest.mark.parametrize('cache_keys', (
      [], ['refs'], ['refs', 'foo'], ['foo', 'bar'],
      ['refs', 'foo', 'bar', 'baz']))
  @pytest.mark.parametrize('parallel', (True, False))
  def test_roundtrip(self, tmpdir, cache_index, cache_keys, parallel):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      size = writer.size
    with granular.DatasetReader(
        directory, granular.decoders, cache_index, cache_keys,
        parallel) as reader:
      assert len(reader) == 10
      assert reader.size == size
      for i in range(10):
        datapoint = reader[i]
        assert datapoint == datapoints[i]

  @pytest.mark.parametrize('cache_index', (True, False))
  @pytest.mark.parametrize('cache_keys', ([], ['refs'], ['refs', 'foo']))
  @pytest.mark.parametrize('parallel', (True, False))
  def test_slicing(self, tmpdir, cache_index, cache_keys, parallel):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'utf8', 'bar': 'int', 'baz': 'utf8[]'}
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
    with granular.DatasetReader(
        directory, granular.decoders, cache_index, cache_keys,
        parallel) as reader:
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

  @pytest.mark.parametrize('cache_index', (True, False))
  @pytest.mark.parametrize('cache_keys', ([], ['refs'], ['refs', 'baz']))
  def test_available(self, tmpdir, cache_index, cache_keys):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      for i in range(10):
        baz = [str(j) for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
    reader = granular.DatasetReader(
        directory, granular.decoders, cache_index, cache_keys)
    with reader:
      for i in range(10):
        available = reader.available(i)
        assert available == {'bar': True, 'baz': range(i), 'foo': True}
        datapoint = reader[i, available]
        assert datapoint == datapoints[i]

  @pytest.mark.parametrize('cache_index', (True, False))
  @pytest.mark.parametrize('cache_keys', ([], ['refs'], ['refs', 'baz']))
  @pytest.mark.parametrize('parallel', (True, False))
  def test_pickle(self, tmpdir, cache_index, cache_keys, parallel):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      size = writer.size
    reader = granular.DatasetReader(
        directory, granular.decoders, cache_index, cache_keys, parallel)
    reader = cloudpickle.loads(cloudpickle.dumps(reader))
    reader = cloudpickle.loads(cloudpickle.dumps(reader))
    with reader:
      assert len(reader) == 10
      assert reader.size == size
      for i in range(10):
        datapoint = reader[i]
        assert datapoint == datapoints[i]

  def test_parallel_exit(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'bar': 'int', 'baz': 'utf8[]', 'foo': 'utf8'}
    datapoints = []
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      for i in range(10):
        baz = [f'word{j}' for j in range(i)]
        datapoint = {'foo': 'hello world', 'bar': i, 'baz': baz}
        writer.append(datapoint)
        datapoints.append(datapoint)
      size = writer.size
    reader = granular.DatasetReader(
        directory, granular.decoders, parallel=True)
    assert len(reader) == 10
    assert reader.size == size
    for i in range(10):
      datapoint = reader[i]
      assert datapoint == datapoints[i]
    # Intentionally omitted reader.close().
