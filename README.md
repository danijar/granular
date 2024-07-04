[![PyPI](https://img.shields.io/pypi/v/granular.svg)](https://pypi.python.org/pypi/granular/#history)

# Granular: Fast format for datasets

Granular is a library for reading and writing multimodal datasets. Each
Granular dataset is a collection of linked files in [bag file format][bag], a
simple seekable container structure. Granular includes a high-performance data
loader that can load from Cloud buckets.

[bag]: ...

## Features

- 🚀 **Performance:** High read and write throughput locally and on Cloud.
- 🔎 **Seeking:** Fast random access from disk by datapoint index.
- 🎞️ **Sequences:** Datapoints can contain seekable lists of modalities.
- 🤸 **Flexibility:** User provides encoders and decoders; examples available.
- 👥 **Sharding:** Store datasets into shards to split processing workloads.
- 🔄 **Determinism:** Deterministic and resumable global shuffling per epoch.

## Installation

```
pip install granular
```

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
    'bar': 'utf8[]',   # *list* of strings
    'baz': 'msgpack',  # packed structure
    'abc': 'jpg',      # image
    'xyz': 'array',    # array
}

with granular.DatasetWriter(directory, spec, granular.encoders) as writer:
  for i in range(10):
    datapoint = {
        'foo': i,
        'bar': ['hello'] * i,
        'baz': {'a': 1},
        'abc': np.zeros((60, 80, 3), np.uint8),
        'xyz': np.arange(0, 1 + i, np.float32),
    }
    writer.append(datapoint)

print(list(directory.glob('*')))
# ['spec.json', 'refs.bag', 'foo.bag', 'bar.bag', 'baz.bag', 'abc.bag', 'xyz.bag']
```

Reading

```python
with granular.DatasetReader(directory, granular.decoders) as reader:
  print(reader.spec)    # {'foo': 'int', 'bar': 'utf8[]', 'baz': 'msgpack', ...}
  print(reader.size)    # Dataset size in bytes
  print(len(reader))    # Number of datapoints

  datapoint = reader[2]
  print(datapoint['foo'])        # 2
  print(datapoint['bar'])        # ['hello', 'hello']
  print(datapoint['abc'].shape)  # (60, 80, 3)
```

Loading

```python
def preproc(datapoint, seed):
  return {'image': datapoint['abc'], 'label': datapoint['foo']}

loader = granular.Loader(
    reader, batch=8, fns=[preproc], shuffle=True, workers=64, seed=0)

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

Retriving a datapoint requires first reading from `refs.bag` to find the
references into the other bag files, and then reading from each of the modality
bag files. If some of the modalities are small enough, they can be cached in
RAM by setting `cache_keys`. In general, it is recommended to cache `refs` as
well as all small modalities, such as integer labels.

Additionally, reading from a Bag file requires two read operations. The first
operation looks at the index table at the end of the file to locate the byte
offset of the record. The second operation retrieves the actual record. In
general, it is recommended to cache the index for all Bag files. Together, the
tables take up `8 * len(spec) * len(reader)` bytes of RAM.

```python
reader = granular.DatasetReader(
    directory, decoders,
    cache_index=True,            # Cache index tables of all bag files in memory.
    cache_keys=('refs', 'foo'),  # Fully cache refs.bag and foo.bag in memory.
)
```

### Masking

It is possible to load the values of only a subset of keys of a datapoint. For
this, provide a mask in addition to the datapoint index. This reduces the
number of read requests to only the bag files that are actually needed:

```python
print(reader.spec)  # {'foo': 'int', 'bar': 'utf8', 'baz': 'array'}

mask = {'foo': True, 'baz': True}
datapoint = reader[index, mask]
print('foo' in datapoint)  # True
print('bar' in datapoint)  # False
print('baz' in datapoint)  # True
```

### Sequences

Each dataset is a list of datapoints. Each datapoint is a dictionary with
string keys and either individual byte values or lists of byte values. To use
sequence values, add the `[]` suffix to the type in the `spec`:

```python
spec = {
    'title': 'utf8',
    'frames': 'jpg[]',
    'captions': 'utf8[]',
    'times': 'int[]',
}
```

Sequence fields can not only store values of variable length, but also allow
reading ranges of the value without loading the whole sequence from disk using
masking:

```python
available = reader.available(index)
print(available)
# {'title': True, 'frames': range(54), 'captions': range(7), 'times': range(7)}

mask = {
    'title': True,            # Read the title modality
    'frames': range(32, 42),  # Read a range of 10 frames.
    'captions': range(0, 7),  # Read all captions.
    'times': True,            # Another way to read the full list.
}
datapoint = reader[index, mask]
print(len(datapoint['frames']))  # 10
```

Ranges are loaded using a single read operation, corresponding to a single
download request on Cloud infrastructure.

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
