[![PyPI](https://img.shields.io/pypi/v/bags.svg)](https://pypi.python.org/pypi/bags/#history)

# 👜 Bags: Fast format for storing datasets

Bags is a library for reading and writing multimodal datasets. Each dataset is
a collection of linked files of the [bag][bag] type, a simple container format.

[bag]: ...

## Features

- **Performance:** Minimal overhead for maximum read and write throughput.
- **Seekable:** Fast random access from disk by datapoint index.
- **Sharding:** Automatically splits large datasets into multiple files.
- **Flexible:** No predefined types, user provides encoders and decoders.
- **Sequences:** Datapoints can reference record range of other bag files.

## Installation

Bags is [a single file][file], so you can just copy it to your project
directory. Or you can install the package:

```
pip install bags
```

[file]: https://github.com/danijar/bags/blob/main/bags/bags.py

## Quickstart

Writing

```python3
import bags
import msgpack
import numpy as np

encoders = {
    'utf8': lambda x: x.encode('utf-8'),
    'int': lambda x, size: x.to_bytes(int(size), 'little'),
    'msgpack': msgpack.packb,
}

spec = {
    'foo': 'int(8)',   # 8-byte integer
    'bar': 'utf8[]',   # list of strings
    'baz': 'msgpack',  # packed structure
}

shardsize = 10 * 1024 ** 3  # 10GB shards

with bags.DatasetWriter(directory, spec, encoders, shardsize) as writer:
  writer.append({'foo': 42, 'bar': ['hello', 'world'], 'baz': {'a': 1})
```

Files

```sh
$ ls directory
spec.json
refs-00001.bag
foo-00001.bag
bar-00001.bag
baz-00001.bag
```

Reading

```python
decoders = {
    'utf8': lambda x: x.decode('utf-8'),
    'int': lambda x, size=None: int.from_bytes(x),
    'msgpack': msgpack.unpackb,
}

with bags.DatasetReader(directory, decoders) as reader:
  print(len(reader))

  # Read data points by index. This will read only the relevant bytes from #
  disk. An additional small read is used when caching index tables is #
  disabled, supporting arbitrarily large datasets with minimal overhead.
  assert reader[0] == {'foo': 42, 'bar': ['hello', 'world'], 'baz': {'a': 1}

  # Read a subset of keys of a datapoint. For example, this allows quickly
  # iterating over the metadata fields of all datapoints without accessing
  # expensive image or video modalities.
  assert reader[0, {'foo': True, 'baz': True}] == {'foo': 42, 'baz': {'a': 1}}

  # Read only a slice of the 'bar' list. Only the requested slice will be
  # fetched from disk. For example, the could be used to load a subsequence of
  a long video that is stored as list of consecutive MP4 clips.
  assert reader[0, {'bar': range(1, 2)}] == {'bar': ['world']}
```

# Serialization

Bags does not impose a serialization solution on the user. Here are examples of
commonly used type strings and corresponding encode and decode functions.

```python
encoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'int': lambda x, size=None: x.to_bytes(
        int(size if size else np.ceil(np.log2(1 + x) / 8))),
    'msgpack': msgpack.packb,
    'array': lambda x, *args: x.tobytes(),
}

decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'int': lambda x, size=None: int.from_bytes(x),
    'msgpack': msgpack.unpackb,
    'array': lambda x, dtype, *shape: np.frombuffer(x, dtype).reshape(
        tuple(int(x) for x in shape))
}
```

Any words can be used as types, as long as an encoder and decoder is available.

Types can be paremeterized with args that will be passed into the encoder and
decoder, such as `array(float32,64,128)`.

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/bags/issues

