[![PyPI](https://img.shields.io/pypi/v/bags.svg)](https://pypi.python.org/pypi/bags/#history)

# Bags: Fast format for datasets

Bags is a library for reading and writing multimodal datasets. Each dataset is
a collection of linked files of the [bag file format][bag] type, a simple
seekable container structure.

[bag]: ...

## Features

- 🚀 **Performance:** Minimal overhead for maximum read and write throughput.
- 🔎 **Seekable:** Fast random access from disk by datapoint index.
- 🎞️ **Sequences:** Datapoints can contain seekable ranges of modalities.
- 🤸 **Flexible:** User provides encoders and decoders; examples available.
- 👥 **Sharding:** Store datasets into shards to split processing workloads.

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

with bags.ShardedDatasetWriter(directory, spec, encoders, shardsize) as writer:
  writer.append({'foo': 42, 'bar': ['hello', 'world'], 'baz': {'a': 1})
  # ...
```

Files

```sh
$ tree directory
.
├── 000000
│  ├── spec.json
│  ├── refs.bag
│  ├── foo.bag
│  ├── bar.bag
│  └── baz.bag
├── 000001
│  ├── spec.json
│  ├── refs.bag
│  ├── foo.bag
│  ├── bar.bag
│  └── baz.bag
└── ...
```

Reading

```python
decoders = {
    'utf8': lambda x: x.decode('utf-8'),
    'int': lambda x, size=None: int.from_bytes(x),
    'msgpack': msgpack.unpackb,
}

with bags.ShardedDatasetReader(directory, decoders) as reader:
  print(len(reader))  # Total number of datapoints.
  print(reader.size)  # Total dataset size in bytes.
  print(reader.shards)

  # Read data points by index. This will read only the relevant bytes from
  # disk. An additional small read is used when caching index tables is
  # disabled, supporting arbitrarily large datasets with minimal overhead.
  assert reader[0] == {'foo': 42, 'bar': ['hello', 'world'], 'baz': {'a': 1}

  # Read a subset of keys of a datapoint. For example, this allows quickly
  # iterating over the metadata fields of all datapoints without accessing
  # expensive image or video modalities.
  assert reader[0, {'foo': True, 'baz': True}] == {'foo': 42, 'baz': {'a': 1}}

  # Read only a slice of the 'bar' list. Only the requested slice will be
  # fetched from disk. For example, the could be used to load a subsequence of
  # a long video that is stored as list of consecutive MP4 clips.
  assert reader[0, {'bar': range(1, 2)}] == {'bar': ['world']}
```

For small datasets where sharding is not necessary, you can also use
`DatasetReader` and `DatasetWriter`. These can also be used to look at
the individual shards of a sharded dataset.

For distributed processing using multiple processes or machines, use
`ShardedDatasetReader` and `ShardedDatasetWriter` and set `shard_start` to the
worker index and `shard_stop` to the total number of workers.

## Formats

Bags does not impose a serialization solution on the user. Any words can be
used as types, as long as an encoder and decoder is provided.

Examples of encode and decode functions for common types are provided in
[formats.py][formats] and include:

- Numpy
- JPEG
- PNG
- MP4

Types can be paremeterized with args that will be forwarded to the encoder and
decoder, for example `array(float32,64,128)`.

[formats]: https://github.com/danijar/bags/blob/main/bags/formats.py

# Bag

The Bag format is a simple container file type. It simply stores a list of byte
blobs, following by an index table of integers for all the start locations in
the file. The start locations are encoded as 8-byte unsigned little-endian and
also include the end offset of the last blob.

This format allows for fast random access, either by loading the index table
into memory upfront, or by doing one small read to find the start and end
locations followed by a targeted large read for the blob content.

Bags builds on top of Bag to read and write datasets of multiple modalities and
where datapoints can contain sequences of blobs of a modality, with efficient
seeking for both datapoints and range queries into modalities.

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/bags/issues

