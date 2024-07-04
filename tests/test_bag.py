import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import granular


class TestBag:

  def test_writer(self, tmpdir):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    total = 8
    with granular.BagWriter(filename) as writer:
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
    with granular.BagReader(filename) as reader:
      reader.size == total

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_roundtrip(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    values = []
    total = 0
    with granular.BagWriter(filename) as writer:
      for i in range(100):
        size = int(rng.integers(4, 100))
        value = int(rng.integers(0, 1000))
        writer.append(value.to_bytes(size, 'little'))
        values.append(value)
        total += size
    with granular.BagReader(filename, cache_index) as reader:
      assert len(reader) == 100
      for index, reference in enumerate(values):
        value = reader[index]
        value = int.from_bytes(value, 'little')
        assert value == reference

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_slicing(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    with granular.BagWriter(filename) as writer:
      for i in range(100):
        writer.append(i.to_bytes(int(rng.integers(4, 32)), 'little'))
    with granular.BagReader(filename, cache_index) as reader:
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

  @pytest.mark.parametrize('cache_index', (True, False))
  def test_pickle(self, tmpdir, cache_index):
    filename = pathlib.Path(tmpdir) / 'file.bag'
    rng = np.random.default_rng(seed=0)
    values = []
    total = 0
    with granular.BagWriter(filename) as writer:
      for i in range(100):
        size = int(rng.integers(4, 100))
        value = int(rng.integers(0, 1000))
        writer.append(value.to_bytes(size, 'little'))
        values.append(value)
        total += size
    reader = granular.BagReader(filename, cache_index)
    with reader:
      [reader[i] for i in range(100)]
    reader = pickle.loads(pickle.dumps(reader))
    reader = pickle.loads(pickle.dumps(reader))
    with reader:
      assert len(reader) == 100
      for index, reference in enumerate(values):
        value = reader[index]
        value = int.from_bytes(value, 'little')
        assert value == reference
