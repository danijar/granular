import pathlib
import pickle

import granular
import numpy as np
import pytest


class TestBag:
    def test_writer(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        idx = pathlib.Path(tmpdir) / 'file.idx'
        rng = np.random.default_rng(seed=0)
        data_total = 0
        with granular.BagWriter(bag) as writer:
            for i in range(100):
                size = rng.integers(4, 100)
                value = i.to_bytes(size, 'little')
                index = writer.append(value)
                assert index == i
                assert len(writer) == i + 1
                data_total += size
            assert writer.size == data_total + 8 * 100
        assert bag.exists()
        assert idx.exists()
        assert bag.stat().st_size == data_total
        assert idx.stat().st_size == 8 * 100
        with granular.BagReader(bag) as reader:
            assert len(reader) == 100
            assert reader.size == data_total + 8 * 100

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize('cache_data', (True, False))
    def test_roundtrip(self, tmpdir, cache_index, cache_data):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        rng = np.random.default_rng(seed=0)
        values = []
        with granular.BagWriter(bag) as writer:
            for i in range(100):
                size = int(rng.integers(4, 100))
                value = int(rng.integers(0, 1000))
                writer.append(value.to_bytes(size, 'little'))
                values.append(value)
        with granular.BagReader(
            bag, cache_index=cache_index, cache_data=cache_data
        ) as reader:
            assert len(reader) == 100
            for index, reference in enumerate(values):
                value = reader[index]
                value = int.from_bytes(value, 'little')
                assert value == reference, index

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize('cache_data', (True, False))
    def test_slicing(self, tmpdir, cache_index, cache_data):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        rng = np.random.default_rng(seed=0)
        with granular.BagWriter(bag) as writer:
            for i in range(100):
                writer.append(i.to_bytes(int(rng.integers(4, 32)), 'little'))
        with granular.BagReader(
            bag, cache_index=cache_index, cache_data=cache_data
        ) as reader:
            assert len(reader) == 100
            for requested in (
                range(0),
                range(1),
                range(10),
                range(3, 5),
                range(90, 100),
                range(90, 110),
            ):
                values = reader[requested]
                values = [int.from_bytes(x, 'little') for x in values]
                expected = [x for x in list(requested) if 0 <= x < 100]
                assert values == expected, requested

    @pytest.mark.parametrize('cache_index', (True, False))
    def test_pickle(self, tmpdir, cache_index):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        rng = np.random.default_rng(seed=0)
        values = []
        with granular.BagWriter(bag) as writer:
            for i in range(100):
                size = int(rng.integers(4, 100))
                value = int(rng.integers(0, 1000))
                writer.append(value.to_bytes(size, 'little'))
                values.append(value)
        reader = granular.BagReader(bag, cache_index=cache_index)
        [reader[i] for i in range(100)]
        reader = pickle.loads(pickle.dumps(reader))
        reader = pickle.loads(pickle.dumps(reader))
        with reader:
            assert len(reader) == 100
            for index, reference in enumerate(values):
                value = reader[index]
                value = int.from_bytes(value, 'little')
                assert value == reference

    def test_read_with_shared_buffers(self, tmpdir):
        from granular.bag import SharedBuffer

        bag = pathlib.Path(tmpdir) / 'file.bag'
        idx = pathlib.Path(tmpdir) / 'file.idx'
        with granular.BagWriter(bag) as writer:
            writer.append(b'hello')
            writer.append(b'world')
        with bag.open('rb') as f:
            bag_data = f.read()
        with idx.open('rb') as f:
            idx_data = f.read()
        bag_buf = SharedBuffer(bag_data)
        idx_buf = SharedBuffer(idx_data)
        with granular.BagReader(bag_buf, idx_buf) as reader:
            assert len(reader) == 2
            assert reader[0] == b'hello'
            assert reader[1] == b'world'

    def test_read_with_bytes(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        idx = pathlib.Path(tmpdir) / 'file.idx'
        with granular.BagWriter(bag) as writer:
            writer.append(b'hello')
            writer.append(b'world')
        with bag.open('rb') as f:
            bag_bytes = f.read()
        with idx.open('rb') as f:
            idx_bytes = f.read()
        with granular.BagReader(bag_bytes, idx_bytes) as reader:
            assert len(reader) == 2
            assert reader[0] == b'hello'
            assert reader[1] == b'world'
