import pathlib
import struct

import granular
import pytest


class TestCompat:
    @pytest.mark.parametrize('cache_data', (True, False))
    @pytest.mark.parametrize('version', (1, 2))
    def test_roundtrip(self, tmpdir, cache_data, version):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        records = [b'hello', b'world', b'test']
        write_bag(bag, records, version)
        with granular.BagReader(
            bag,
            cache_data=cache_data,
            single_file=True,
        ) as reader:
            assert len(reader) == 3
            assert reader[0] == b'hello'
            assert reader[1] == b'world'
            assert reader[2] == b'test'

    @pytest.mark.parametrize('version', (1, 2))
    def test_slicing(self, tmpdir, version):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        records = [i.to_bytes(4, 'little') for i in range(100)]
        write_bag(bag, records, version)
        with granular.BagReader(bag, single_file=True) as reader:
            values = reader[range(10, 20)]
            expected = [i.to_bytes(4, 'little') for i in range(10, 20)]
            assert values == expected


def write_bag(path, records, version):
    limst = struct.Struct('<Q')
    assert version in (1, 2), version
    with path.open('wb') as f:
        offsets = [0]
        for record in records:
            f.write(record)
            offsets.append(offsets[-1] + len(record))
        if version == 2:
            offsets = offsets[1:]
        for offset in offsets:
            f.write(limst.pack(offset))
