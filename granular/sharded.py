import concurrent.futures
import itertools
import operator
import pathlib
import pickle

from . import utils
from . import dataset


class ShardedDatasetWriter(utils.Closing):
    def __init__(
        self,
        directory,
        spec,
        encoders,
        shardlen=None,
        shardstart=0,
        shardstep=1,
    ):
        super().__init__()
        assert 0 <= shardstart
        assert 1 <= shardstep
        if shardstart > 0 or shardstep > 1:
            assert shardlen, shardlen
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        self.directory = directory
        self.thespec = spec
        self.encoders = encoders
        self.shardlength = shardlen
        self.shardstart = shardstart
        self.shardnum = shardstart
        self.shardstep = shardstep
        self.prevshards = 0
        self.prevsize = 0
        self.prevlength = 0
        self.writer = None

    @property
    def spec(self):
        return self.thespec

    @property
    def shards(self):
        return self.prevshards + bool(self.writer)

    @property
    def size(self):
        return self.prevsize + (self.writer.size if self.writer else 0)

    def __len__(self):
        return self.prevlength + (len(self.writer) if self.writer else 0)

    def append(self, datapoint, flush=True):
        if not self.writer:
            folder = self.directory / f'{self.shardnum:06}'
            self.writer = dataset.DatasetWriter(
                folder, self.spec, self.encoders
            )
        self.writer.append(datapoint)
        flush and self.writer.flush()
        if self.shardlength and len(self.writer) >= self.shardlength:
            self.prevshards += 1
            self.prevsize += self.writer.size
            self.prevlength += len(self.writer)
            self.writer.close()
            self.writer = None
            self.shardnum += self.shardstep
        return len(self) - 1

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


class ShardedDatasetReader(utils.Closing):
    def __init__(
        self,
        directory,
        decoders,
        cache_index=True,
        cache_keys=(),
        parallel=False,
        shardstart=0,
        shardstep=1,
    ):
        super().__init__()
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        folders = sorted(directory.glob('*'))
        assert all(int(x.name) == i for i, x in enumerate(folders)), folders
        range_ = range(shardstart, len(folders), shardstep)
        selected = [folders[i] for i in range_]
        assert selected, (folders, selected, shardstart, shardstep)
        args = (decoders, cache_index, cache_keys, parallel)
        if parallel:
            make = lambda x: dataset.DatasetReader(x, *args)
            with concurrent.futures.ThreadPoolExecutor(len(selected)) as pool:
                self.readers = list(pool.map(make, selected))
        else:
            self.readers = [dataset.DatasetReader(x, *args) for x in selected]
        lengths = [len(x) for x in self.readers]
        self.stops = list(itertools.accumulate(lengths, operator.add))
        self.starts = [0] + self.stops[:-1]
        self.length = sum(lengths)

    @property
    def spec(self):
        return self.readers[0].spec

    @property
    def size(self):
        return sum(x.size for x in self.readers)

    @property
    def shards(self):
        return len(self.readers)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, keys = index
        else:
            keys = tuple(self.spec.keys())
        if isinstance(index, int):
            readers, local_indices = self._resolve(index, index + 1)
            assert len(readers) == len(local_indices) == 1
            return readers[0][local_indices[0][0], keys]
        else:
            # We could parallelize this, but typically the requested slice
            # touches only either one or two shards. A thread pool would make
            # it harder to control the maximum number of concurrent
            # connections.
            assert index.step in (None, 1), index
            results = []
            resolved = self._resolve(index.start, index.stop)
            for reader, local_index in zip(*resolved):
                results.append(reader[local_index, keys])
            return {k: sum([v[k] for v in results], []) for k in results[0]}

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def close(self):
        for reader in self.readers:
            reader.close()

    def _resolve(self, start, stop):
        assert 0 <= start <= stop <= len(self), (start, stop, len(self))
        readers, local_indices = [], []
        for reader, left, right in zip(self.readers, self.starts, self.stops):
            if not (start < right):
                continue
            elif stop <= right:
                readers.append(reader)
                local_indices.append(range(start - left, stop - left))
                break
            else:
                readers.append(reader)
                local_indices.append(range(start - left, right - left))
                start = right
        return readers, local_indices
