import concurrent.futures
import json
import pathlib
import pickle
import re

from . import utils
from . import bag


class DatasetWriter(utils.Closing):
    def __init__(self, directory, spec, encoders):
        super().__init__()
        assert all(re.match(r'^[a-z_]', k) for k in spec.keys()), spec
        assert all(re.match(r'^[a-z_]', v) for v in spec.values()), spec
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        assert not directory.exists(), directory
        directory.mkdir()
        spec = dict(sorted(spec.items(), key=lambda x: x[0]))
        if encoders is None:
            encoders = {k: None for k in spec.keys()}
        else:
            encoders = {k: encoders[v] for k, v in spec.items()}
        self.directory = directory
        self.encoders = encoders
        self.thespec = spec
        self.writers = {
            k: bag.BagWriter(self.directory / f'{k}.bag')
            for k in self.spec.keys()
        }
        self.specwritten = False
        self.length = 0

    @property
    def spec(self):
        return self.thespec

    @property
    def size(self):
        return sum(x.size for x in self.writers.values())

    def __len__(self):
        return self.length

    def append(self, datapoint, flush=True):
        assert isinstance(datapoint, dict)
        assert set(datapoint.keys()) == set(self.spec.keys()), (
            datapoint.keys(),
            self.spec,
        )
        for key, dtype in self.spec.items():
            writer = self.writers[key]
            value = datapoint[key]
            value = self._encode(key, value, dtype)
            writer.append(value, flush=False)
        index = self.length
        self.length += 1
        flush and self.flush()
        return index

    def flush(self):
        if not self.specwritten:
            self.specwritten = True
            content = json.dumps(self.spec).encode('utf-8')
            with (self.directory / 'spec.json').open('wb') as f:
                f.write(content)
        for writer in self.writers.values():
            writer.flush()

    def close(self):
        self.flush()
        for writer in self.writers.values():
            writer.close()

    def _encode(self, key, value, dtype):
        encoder = self.encoders[key]
        if not encoder:
            return value
        try:
            value = encoder(value)
            assert isinstance(value, bytes), (key, type(value))
            return value
        except Exception:
            print(f"Error encoding key '{key}' of type '{dtype}'.")
            raise


class DatasetReader(utils.Closing):
    def __init__(
        self,
        directory,
        decoders,
        cache_index=True,
        cache_keys=(),
        parallel=False,
    ):
        super().__init__()
        assert isinstance(cache_keys, (tuple, list)), cache_keys
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        with (directory / 'spec.json').open('rb') as f:
            self.thespec = json.loads(f.read())
        info = (self.thespec, cache_keys)
        assert all(k in self.thespec for k in cache_keys), info
        self.readers = {
            k: bag.BagReader(
                directory / f'{k}.bag',
                cache_index=cache_index,
                cache_data=(k in cache_keys),
            )
            for k in self.spec.keys()
        }
        lengths = {k: len(v) for k, v in self.readers.items()}
        assert len(set(lengths.values())) == 1, (directory, lengths)
        self.length, = set(lengths.values())
        if decoders is None:
            decoders = {k: None for k in self.spec.keys()}
        else:
            decoders = {k: decoders[v] for k, v in self.spec.items()}
        self.decoders = decoders
        self.cache_keys = cache_keys
        self.workers = int(parallel and len(set(self.spec) - set(cache_keys)))
        if self.workers:
            self.pool = concurrent.futures.ThreadPoolExecutor(self.workers)

    @property
    def spec(self):
        return self.thespec

    @property
    def size(self):
        return sum(x.size for x in self.readers.values())

    def __getstate__(self):
        d = self.__dict__.copy()
        if self.workers:
            d.pop('pool')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.workers:
            self.pool = concurrent.futures.ThreadPoolExecutor(self.workers)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, keys = index
            if isinstance(keys, list):
                keys = tuple(keys)
            assert isinstance(keys, tuple), keys
        else:
            keys = tuple(self.spec.keys())
        assert all(k in self.spec for k in keys), (self.spec, keys)
        if isinstance(index, int):
            point = self._fetch(keys, index)
            decoded = {
                k: self._decode(k, v, self.spec[k]) for k, v in point.items()
            }
        else:
            i = index
            assert 0 <= i.start <= i.stop and i.step in (None, 1), i
            points = self._fetch(keys, index)
            decoded = {
                k: [self._decode(k, v, self.spec[k]) for v in vs]
                for k, vs in points.items()
            }
        return decoded

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def close(self):
        if self.workers:
            self.pool.shutdown(wait=False)
        for reader in self.readers.values():
            reader.close()

    def _fetch(self, keys, index):
        if self.workers:
            cached = lambda k: k in self.cache_keys
            reqs1 = [k for k in keys if not cached(k)]
            reqs2 = [k for k in keys if cached(k)]
            fn = lambda k: self.readers[k][index]
            values1 = self.pool.map(fn, reqs1)
            values2 = [self.readers[k][index] for k in reqs2]
            datapoint = dict([*zip(reqs1, values1), *zip(reqs2, values2)])
        else:
            datapoint = {k: self.readers[k][index] for k in keys}
        return datapoint

    def _decode(self, key, value, dtype):
        decoder = self.decoders[key]
        if not decoder:
            return value
        try:
            if isinstance(value, list):
                return [decoder(x) for x in value]
            else:
                return decoder(value)
        except Exception:
            print(f"Error decoding key '{key}' of type '{dtype}'.")
            raise
