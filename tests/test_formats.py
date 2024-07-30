import pathlib

import granular
import numpy as np


class TestFormats:

  def test_formats(self, tmpdir):
    directory = pathlib.Path(tmpdir) / 'dataset'
    spec = {
        'a': 'utf8',
        'b': 'int',
        'c': 'utf8[]',
        'd': 'msgpack',
        'e': 'int[]',
        'f': 'bytes',
        'g': 'array',
        'h': 'jpg',
        'i': 'png',
        'j': 'mp4',
        'k': 'tree',
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
          'k': [{'a': [np.ones(5), np.zeros(2)]}, np.ones(3), 'hello'],
      })
    with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
      assert writer.spec == spec
      for datapoint in datapoints:
        writer.append(datapoint)
    with granular.DatasetReader(directory, granular.decoders) as reader:
      assert reader.spec == spec
      for i in range(len(reader)):
        actual = reader[i]
        reference = datapoints[i]
        assert tree_equals(actual, reference)


def tree_equals(xs, ys):
  assert type(xs) == type(ys)
  if isinstance(xs, (list, tuple)):
    assert len(xs) == len(ys)
    return all(tree_equals(x, y) for x, y in zip(xs, ys))
  elif isinstance(xs, dict):
    assert xs.keys() == ys.keys()
    return all(tree_equals(xs[k], ys[k]) for k in xs.keys())
  elif isinstance(xs, np.ndarray):
    return (xs == ys).all()
  else:
    return xs == ys
