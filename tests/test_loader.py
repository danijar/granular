import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import granular


class TestLoader:

  @pytest.mark.parametrize('batch', (1, 3, 6, 12))
  def test_ordered(self, tmpdir, batch):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'int', 'bar': 'array'}
    datapoints = [{'foo': i, 'bar': np.ones((3, 3))} for i in range(10)]
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      [writer.append(x) for x in datapoints]
    source = granular.DatasetReader(directory, granular.decoders)
    dataset = iter(granular.Loader(source, batch, shuffle=False, workers=4))
    for i in range(0, 2 * len(datapoints), batch):
      data = next(dataset)
      assert set(data.keys()) == {'foo', 'bar'}
      assert data['foo'].shape == (batch,)
      assert data['bar'].shape == (batch, 3, 3)
      assert (data['foo'] == (
          np.arange(i, i + batch) % len(datapoints))).all()
      assert (data['bar'] == np.ones((batch, 3, 3))).all()

  @pytest.mark.parametrize('batch', (1, 3))
  def test_shuffled(self, tmpdir, batch):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'int', 'bar': 'array'}
    datapoints = [{'foo': i, 'bar': np.ones((3, 3))} for i in range(10)]
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      [writer.append(x) for x in datapoints]
    source = granular.DatasetReader(directory, granular.decoders)
    dataset = iter(granular.Loader(source, batch, shuffle=True, workers=4))
    seen = set()
    for i in range(0, len(datapoints), batch):
      data = next(dataset)
      assert (data['bar'] == np.ones((batch, 3, 3))).all()
      if i + batch <= len(datapoints):
        assert all([x not in seen for x in data['foo']])
        seen.update(data['foo'])
      else:
        seen.clear()

  @pytest.mark.parametrize('kwargs', (
      dict(cache_index=False, cache_keys=[], parallel=False),
      dict(cache_index=True, cache_keys=['refs'], parallel=True),
  ))
  def test_reader_options(self, tmpdir, kwargs, batch=3):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'int', 'bar': 'array'}
    datapoints = [{'foo': i, 'bar': np.ones((3, 3))} for i in range(10)]
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      [writer.append(x) for x in datapoints]
    source = granular.DatasetReader(directory, granular.decoders, **kwargs)
    dataset = iter(granular.Loader(source, batch, shuffle=True, workers=4))
    seen = set()
    for i in range(0, len(datapoints), batch):
      data = next(dataset)
      assert (data['bar'] == np.ones((batch, 3, 3))).all()
      if i + batch <= len(datapoints):
        assert all([x not in seen for x in data['foo']])
        seen.update(data['foo'])
      else:
        seen.clear()
