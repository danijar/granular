[![PyPI](https://img.shields.io/pypi/v/granular.svg)](https://pypi.python.org/pypi/granular/#history)

# Granular

Granular is simple and scalable dataset format. Each dataset is a collection
of seekable record files that support fast random accesses and resumable
appends. Granular comes with a high-performance data loader.

```
pip install granular
```

**NOTE:** The API and file format have been updated in version 0.22.0 to
support resumable writes. Previously written datasets can still be read. Pin
`granular<=0.21.2` to continue using the [previous version][previous].

[previous]: https://github.com/danijar/granular/tree/087bc1c529aa7716bdf56d1a0edf0175ab983325

## Features

- 🚀 **Performance:** High read and write throughput locally and on Cloud.
- 🔎 **Seeking:** Fast random access from disk by datapoint index.
- 🤸 **Flexibility:** User provides encoders and decoders; examples available.
- 👥 **Sharding:** Store datasets into shards to split processing workloads.
- 🔄 **Determinism:** Deterministic and resumable global shuffling per epoch.
- ✅ **Correctness:** A suite of unit tests with high code coverage.

## Quickstart

```python3
import pathlib
import granular
import numpy as np

directory = './dataset'
```

Writing

```python
spec = {
    'foo': 'int',      # integer
    'bar': 'utf8',     # string
    'baz': 'msgpack',  # packed structure
    'abc': 'jpg',      # image
    'xyz': 'array',    # array
}

with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
  for i in range(10):
    datapoint = {
        'foo': i,
        'bar': 'hello',
        'baz': {'a': 1},
        'abc': np.zeros((60, 80, 3), np.uint8),
        'xyz': np.arange(0, 1 + i, np.float32),
    }
    writer.append(datapoint)

print(list(directory.glob('*')))
# 'spec.json', 'foo.bag', 'foo.idx', 'bar.bag', 'bar.idx', ...]
```

Reading

```python
with granular.DatasetReader(directory, granular.decoders) as reader:
  print(reader.spec)    # {'foo': 'int', 'bar': 'utf8', 'baz': 'msgpack', ...}
  print(reader.size)    # Dataset size in bytes
  print(len(reader))    # Number of datapoints

  datapoint = reader[2]
  print(datapoint['foo'])        # 2
  print(datapoint['bar'])        # 'hello'
  print(datapoint['abc'].shape)  # (60, 80, 3)
```

Loading

```python
def preproc(datapoint, seed):
  return {'image': datapoint['abc'], 'label': datapoint['foo']}

source = granular.sources.Epochs(reader, shuffle=True, seed=0)
source = granular.sources.Transform(source, preproc)
loader = granular.Loader(source, batch=8, workers=32)

print(loader.spec)
# {'image': (np.uint8, (60, 80, 3)), 'label': (np.int64, ())}

dataset = iter(loader)
for _ in range(100):
  batch = next(dataset)
  print(batch['image'].shape)  # (8, 60, 80, 3)
```

## Advanced

### Filesystems

Custom filesystems are supported by providing different `Path` implementations.
For example, on Google Cloud you can use the `Path` from [elements][elements]
that is optimized for data loading throughput:

```python
import elements  # pip install elements

directory = elements.Path('gs://<bucket>/dataset')

reader = granular.DatasetReader(directory, ...)
wrtier = granular.DatasetWriter(directory, ...)
```

[elements]: https://github.com/danijar/elements

### Formats

Granular does not impose a serialization solution on the user. Any strings can
be used as types in `spec`, as long as their encoder and decoder functions are
provided, for example:

```python
import msgpack

encoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'msgpack': msgpack.packb,
}

decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'msgpack': msgpack.unpackb,
}
```

Examples of common encode and decode functions are provided in
[formats.py][formats]. These support Numpy arrays, images, videos, and more.
They can be used as `granular.encoders` and `granular.decoders`.

[formats]: https://github.com/danijar/granular/blob/main/granular/formats.py

### Resuming

The dataloader is fully deterministic and resumable, given only the step and
seed integers. For this, checkpoint the state dictionary returned by
`loader.save()` and pass this into `loader.load()` when storing a checkpoint.

```python
state = loader.save()
print(state)  # {'step': 100, 'seed': 0}
loader.load(state)
```

### Caching

If some keys have small enough values, they can be cached in RAM by setting
`cache_keys`. It is recommended to cache all small values, such as integer
labels.

Additionally, reading from a Bag file requires two read operations. The first
operation looks at the index file (`.idx`) to locate the byte offset of the
record. The second operation retrieves the actual record from the data file
(`.bag`). It is recommended to cache the index for all Bag files. Together, the
index files take up `8 * len(spec) * len(reader)` bytes of RAM.

```python
reader = granular.DatasetReader(
    directory, decoders,
    cache_index=True,     # Cache index tables of all bag files in memory.
    cache_keys=('foo',),  # Fully cache foo.bag in memory.
)
```

### Columns

It is possible to load the values of only a subset of keys of a datapoint. For
this, provide a tuple of keys in addition to the datapoint index. This reduces
the number of read requests to only the bag files that are actually needed:

```python
print(reader.spec)  # {'foo': 'int', 'bar': 'utf8', 'baz': 'array'}

keys = ('foo', 'baz')
datapoint = reader[index, keys]
print('foo' in datapoint)  # True
print('bar' in datapoint)  # False
print('baz' in datapoint)  # True
```

### Sharding

Large datasets can be stored as list of smaller datasets to easily parallelize
processing, by processing each smaller dataset individually in a different
process or on a different machine. The shard length specifies the number of
datapoints per shard. A good default is to set the number of datapoints such
that each shard is around 10 Gb in size.

```python
# Write into a sharded dataset.
writer = granular.ShardedDatasetWriter(directory, spec, encoders, shardlen=10000)

# Read from a sharded dataset.
reader = granular.ShardedDatasetReader(directory, decoders)
```

The file structure of a sharded dataset is one folder per shard, named after
the shard number. Each shard itself is a dataset and can also be read using the
non-sharded `granular.DatasetReader`.

```sh
$ tree ./directory
.
├── 000000
│   ├── spec.json
│   ├── foo.bag
│   ├── foo.idx
│   ├── bar.bag
│   ├── bar.idx
│   └── ...
├── 000001
│   └── ...
└── ...
```

When processing a dataset with a large number of shards using a smaller number
of workers, specify `shardstart` and `shardstep` so each worker reads and
writes its dedicated subset of shards.

```python
# Write into a sharded dataset.
writer = granular.ShardedDatasetWriter(
    directory, spec, encoders, shardlen=10000,
    shardstart=worker_id,   # Start writing at this shard.
    shardstep=num_workers,  # Afterwards, jump this many shards ahead.
)

# Read from a sharded dataset.
reader = granular.ShardedDatasetReader(
    directory, decoders,
    shardstart=worker_id,   # Start reading at this shard.
    shardstep=num_workers,  # Afterwards, jump this many shards ahead.
)
```

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/granular/issues
