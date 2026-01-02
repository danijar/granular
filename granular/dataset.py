import collections
import concurrent.futures
import json
import pathlib
import pickle
import re
import struct

from . import utils
from . import bag


class DatasetWriter(utils.Closing):
    def __init__(self, directory, spec, encoders):
        super().__init__()
        assert all(re.match(r'^[a-z_]', k) for k in spec.keys()), spec
        assert all(re.match(r'^[a-z_]', v) for v in spec.values()), spec
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        if encoders is None:
            encoders = collections.defaultdict(lambda: None)
        spec = dict(sorted(spec.items(), key=lambda x: x[0]))
        self.directory = directory
        self.encoders = {k: encoders[v] for k, v in spec.items()}
        self.thespec = spec
        self._writespec()
        self.writers = {
            k: bag.BagWriter(self.directory / f'{k}.bag')
            for k in self.spec.keys()
        }
        # Some columns might be ahead from preemption.
        self.length = min(len(x) for x in self.writers.values())

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
            if len(writer) > self.length:
                # Column is ahead from preemption, verify record matches.
                self._verify(key, self.length, value)
            else:
                writer.append(value, flush=False)
        index = self.length
        self.length += 1
        flush and self.flush()
        return index

    def flush(self):
        for writer in self.writers.values():
            writer.flush()

    def close(self):
        self.flush()
        for writer in self.writers.values():
            writer.close()

    def _writespec(self):
        filename = self.directory / 'spec.json'
        if filename.exists():
            existing = json.loads(filename.read_bytes())
            assert self.spec == existing, (self.spec, existing)
        else:
            self.directory.mkdir(exist_ok=True, parents=True)
            filename.write_bytes(json.dumps(self.spec).encode('utf-8'))

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

    def _verify(self, key, index, expected):
        path = self.directory / f'{key}.bag'
        idx_path = self.directory / f'{key}.idx'
        with idx_path.open('rb') as f:
            if index == 0:
                start = 0
            else:
                f.seek((index - 1) * 8)
                start = struct.unpack('<Q', f.read(8))[0]
            f.seek(index * 8)
            end = struct.unpack('<Q', f.read(8))[0]
        with path.open('rb') as f:
            f.seek(start)
            existing = f.read(end - start)
        if existing != expected:
            raise ValueError(
                f"Record mismatch in column '{key}' at index {index}: "
                f'existing {len(existing)} bytes != new {len(expected)} bytes.'
            )


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
        if decoders is None:
            decoders = collections.defaultdict(lambda: None)
        self.thespec = json.loads((directory / 'spec.json').read_bytes())
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
        msg = f'Inconsistent column lengths: {lengths}'
        assert len(set(lengths.values())) == 1, msg
        (self.length,) = set(lengths.values())
        self.decoders = {k: decoders[v] for k, v in self.spec.items()}
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
        dec = lambda k, v: self._decode(k, v, self.spec[k])
        if isinstance(index, int):
            point = self._fetch(keys, index)
            decoded = {k: dec(k, v) for k, v in point.items()}
        else:
            i = index
            assert 0 <= i.start <= i.stop and i.step in (None, 1), i
            points = self._fetch(keys, index)
            decoded = {k: [dec(k, v) for v in vs] for k, vs in points.items()}
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
