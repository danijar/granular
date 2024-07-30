import multiprocessing
import pathlib
import queue

import granular
import numpy as np
import pytest


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
    dataset.close()
    source.close()

  @pytest.mark.parametrize('recycle_after', (False, 1, 2))
  def test_recycle(self, tmpdir, recycle_after, batch=3):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'int', 'bar': 'array'}
    datapoints = [{'foo': i, 'bar': np.ones((3, 3))} for i in range(10)]
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      [writer.append(x) for x in datapoints]
    source = granular.DatasetReader(directory, granular.decoders)
    dataset = iter(granular.Loader(
        source, batch, shuffle=False, workers=4, recycle_after=recycle_after))
    for i in range(0, 2 * len(datapoints), batch):
      data = next(dataset)
      assert set(data.keys()) == {'foo', 'bar'}
      assert data['foo'].shape == (batch,)
      assert data['bar'].shape == (batch, 3, 3)
      assert (data['foo'] == (
          np.arange(i, i + batch) % len(datapoints))).all()
      assert (data['bar'] == np.ones((batch, 3, 3))).all()
    dataset.close()
    source.close()

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
    dataset.close()
    source.close()

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
    dataset.close()
    source.close()

  def test_shared_array_pool(self):
    import granular.loader
    mp = multiprocessing.get_context('spawn')
    array = granular.loader.SharedArray((10, 8), np.int32)
    with mp.Pool(4) as pool:
      list(pool.map(fill1, ((array, i) for i in reversed(range(10)))))
    result = array.result()
    assert result.shape == (10, 8)
    assert (result == np.arange(10)[:, None]).all()

  def test_shared_array_queue(self):
    import granular.loader
    mp = multiprocessing.get_context('spawn')
    array = granular.loader.SharedArray((10, 8), np.int32)
    stop = mp.Event()
    iqueue = mp.Queue()
    oqueue = mp.Queue()
    args = (stop, iqueue, oqueue)
    workers = [mp.Process(target=fill2, args=args) for _ in range(4)]
    [x.start() for x in workers]
    for i in reversed(range(10)):
      iqueue.put((array, i))
    for i in reversed(range(10)):
      oqueue.get()
    stop.set()
    [x.join() for x in workers]
    result = array.result()
    assert result.shape == (10, 8)
    assert (result == np.arange(10)[:, None]).all()

  @pytest.mark.parametrize('running', (True, False))
  def test_checkpoint(self, tmpdir, running, batch=2):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {'foo': 'int'}
    datapoints = [{'foo': i} for i in range(10)]
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      [writer.append(x) for x in datapoints]
    source = granular.DatasetReader(directory, granular.decoders)
    dataset = iter(granular.Loader(source, batch, shuffle=False, workers=4))
    for i in range(0, 2 * batch, batch):
      assert np.all(next(dataset)['foo'] == np.arange(i, i + batch) % 10)
    assert dataset.consumed == 2 * batch
    state = dataset.save()
    dataset.close()
    source.close()
    del dataset
    del source
    source = granular.DatasetReader(directory, granular.decoders)
    dataset = granular.Loader(source, batch, shuffle=False, workers=4)
    if running:
      dataset = iter(dataset)
      dataset.load(state)
    else:
      dataset.load(state)
      dataset = iter(dataset)
    assert dataset.consumed == 2 * batch
    for i in range(2 * batch, 6 * batch, batch):
      assert np.all(next(dataset)['foo'] == np.arange(i, i + batch) % 10)
    assert dataset.consumed == 6 * batch
    dataset.close()
    source.close()


def fill1(args):
  target, i = args
  target.array[i] = i


def fill2(stop, iqueue, oqueue):
  while not stop.is_set():
    try:
      message = iqueue.get(timeout=0.1)
    except queue.Empty:
      continue
    target, i = message
    target.array[i] = i
    oqueue.put(i)
