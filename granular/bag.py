import functools
import os
import pathlib
import pickle
import struct
from multiprocessing import shared_memory

from . import utils


limst = struct.Struct('<Q')


class BagWriter(utils.Closing):
    def __init__(self, bag_path, idx_path=None):
        super().__init__()
        if isinstance(bag_path, str):
            bag_path = pathlib.Path(bag_path)
        if isinstance(idx_path, str):
            idx_path = pathlib.Path(idx_path)
        if idx_path is None:
            idx_path = bag_path.with_suffix('.idx')
        self.bag_path = bag_path
        self.idx_path = idx_path
        if self.idx_path.exists():
            self.length, self.offset = self._resume()
        else:
            self.length, self.offset = 0, 0
        self.bag = self.bag_path.open('ab')
        self.idx = self.idx_path.open('ab')
        self.towrite = []

    @property
    def size(self):
        return self.offset + 8 * self.length

    def __len__(self):
        return self.length

    def append(self, record, flush=True):
        assert len(record), 'empty record'
        assert not self.closed
        assert self.length < 2**32 - 1, self.length
        assert isinstance(record, bytes), type(record)
        index = self.length
        self.length += 1
        self.towrite.append(record)
        flush and self.flush()
        return index

    def flush(self):
        assert not self.closed
        if not self.towrite:
            return
        combined = b''.join(self.towrite)
        if self.bag.tell() > self.offset:  # Resume after partial write
            combined = self._skip(combined)
        if combined:
            self.bag.write(combined)
            self.bag.flush()
        entries = []
        for record in self.towrite:
            self.offset += len(record)
            entries.append(limst.pack(self.offset))
        self.idx.write(b''.join(entries))
        self.idx.flush()
        self.towrite.clear()

    def close(self):
        assert not self.closed
        self.flush()
        self.bag.close()
        self.idx.close()

    def _resume(self):
        if not self.idx_path.exists():
            return 0, 0
        with self.idx_path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            length = f.tell() // 8
            if length == 0:
                offset = 0
            else:
                f.seek((length - 1) * 8)
                offset = limst.unpack(f.read(8))[0]
        return length, offset

    def _skip(self, data):
        with self.bag_path.open('rb') as f:
            f.seek(self.offset)
            existing = f.read(len(data))
        skip = min(len(existing), len(data))
        if data[:skip] != existing[:skip]:
            raise ValueError(
                'Cannot resume after partial write because new and existing '
                'data does not match ({skip} bytes).'
            )
        return data[skip:]


class BagReader(utils.Closing):
    def __init__(
        self,
        bag_source,
        idx_source=None,
        cache_index=True,
        cache_data=False,
        single_file=False,  # Old format
    ):
        super().__init__()

        # User inputs
        if isinstance(bag_source, str):
            bag_source = pathlib.Path(bag_source)
        if isinstance(idx_source, str):
            idx_source = pathlib.Path(idx_source)
        if idx_source is None and not single_file:
            msg = 'can only infer idx_source if bag_source is a path'
            assert ispath(bag_source), msg
            idx_source = bag_source.with_suffix('.idx')

        # Bag source
        if cache_data and ispath(bag_source):
            with bag_source.open('rb') as f:
                bag_source = SharedBuffer(f.read())
        if not hasattr(bag_source, 'open'):
            bag_source = SharedBuffer(bag_source)
        assert hasattr(bag_source, 'open'), 'should be path or shared buffer'
        self.bag_source = bag_source

        # Extract index
        if single_file:
            assert idx_source is None
            assert cache_index is True
            cache_index = False
            self.bag.seek(0, os.SEEK_END)
            filesize = self.bag.tell()
            if filesize < 8:
                idx_source = SharedBuffer(b'')
            else:
                self.bag.seek(-8, os.SEEK_END)
                indexstart = limst.unpack(self.bag.read(8))[0]
                self.bag.seek(indexstart, os.SEEK_SET)
                if limst.unpack(self.bag.read(8))[0] == 0:
                    indexstart += 8  # Optional zero address
                length = (filesize - indexstart) // 8
                self.bag.seek(indexstart, os.SEEK_SET)
                idx_source = SharedBuffer(self.bag.read(length * 8))

        # Index source
        if cache_index and ispath(idx_source):
            with idx_source.open('rb') as f:
                idx_source = SharedBuffer(f.read())
        if not hasattr(idx_source, 'open'):
            cache_index = False
            idx_source = SharedBuffer(idx_source)
        self.idx_source = idx_source
        self.idx.seek(0, os.SEEK_END)
        self.length = self.idx.tell() // 8

    @functools.cached_property
    def bag(self):
        return self.bag_source.open('rb')

    @functools.cached_property
    def idx(self):
        return self.idx_source.open('rb')

    @property
    def size(self):
        if self.length == 0:
            return 0
        _, datasize = self._getlim(self.length - 1, self.length)
        return datasize + self.length * 8

    def __getstate__(self):
        return {
            'bag_source': self.bag_source,
            'idx_source': self.idx_source,
            'length': self.length,
        }

    def __setstate__(self, d):
        self.bag_source = d['bag_source']
        self.idx_source = d['idx_source']
        self.length = d['length']
        self.closed = False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert not self.closed
        assert isinstance(index, (int, slice, range)), (index, type(index))
        if isinstance(index, int):
            lhs, rhs = self._getlim(index, index + 1)
            self.bag.seek(lhs, os.SEEK_SET)
            return self.bag.read(rhs - lhs)
        i = index
        assert 0 <= i.start <= i.stop and i.step in (None, 1), i
        index = range(index.start, min(index.stop, self.length))
        if index.start == index.stop:
            return []
        limits = self._getlim(index.start, index.stop)
        lhs, rhs = limits[0], limits[-1]
        self.bag.seek(lhs, os.SEEK_SET)
        buffer = self.bag.read(rhs - lhs)
        pairs = zip(limits[:-1], limits[1:])
        records = [buffer[i - lhs : j - lhs] for i, j in pairs]
        return records

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def close(self):
        assert not self.closed
        self.bag.close()
        self.idx.close()
        if isinstance(self.idx_source, SharedBuffer):
            if not self.idx_source.closed:
                self.idx_source.close()
        if isinstance(self.bag_source, SharedBuffer):
            if not self.bag_source.closed:
                self.bag_source.close()

    def _getlim(self, start, stop):
        assert start < stop, (start, stop)
        lhs = 8 * max(0, start - 1)
        rhs = 8 * stop
        self.idx.seek(lhs, os.SEEK_SET)
        buffer = self.idx.read(rhs - lhs)
        limits = [
            limst.unpack(buffer[i : i + 8])[0]
            for i in range(0, len(buffer), 8)
        ]
        return limits if start else (0, *limits)


class SharedBuffer:
    ENABLE = True

    def __init__(self, content):
        if self.ENABLE:
            self.size = len(content)
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)
            # The slice range is needed on MacOS where SharedMemory buffers can
            # be larger than requested.
            self.shm.buf[: self.size] = memoryview(content)
            self.buf = self.shm.buf
        else:
            self.buf = bytes(content)

    @property
    def closed(self):
        return self.buf is None

    def __getitem__(self, index):
        return self.buf[index]

    def __getstate__(self):
        if self.ENABLE:
            return (self.shm.name, self.size)
        else:
            return self.buf

    def __setstate__(self, value):
        if self.ENABLE:
            name, size = value
            self.shm = shared_memory.SharedMemory(name=name)
            self.buf = self.shm.buf
            self.size = size
        else:
            self.buf = value

    def open(self, mode='rb', offset=0):
        assert mode in ('rb', 'wb'), mode
        return BufferView(self.buf, self.size, offset)

    def close(self):
        self.buf = None
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class BufferView:
    def __init__(self, buf, size, offset=0):
        self.buf = buf
        self.size = size
        self.offset = offset
        self.pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset - self.offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos + self.offset

    def read(self, n=-1):
        buf = self.buf
        size = self.size
        if n < 0:
            result = buf[self.pos : size]
            self.pos = size
        else:
            result = buf[self.pos : self.pos + n]
            self.pos += n
        return bytes(result)

    def tell(self):
        return self.pos + self.offset

    def close(self):
        self.buf = None


def ispath(obj):
    attrs = ('open', 'parent', 'name', 'with_suffix')
    return all(hasattr(obj, name) for name in attrs)
