import pathlib

import granular
import pytest


class TestCommit:
    def test_commit(self, tmpdir):
        root = pathlib.Path(tmpdir)
        spec = {'foo': 'int'}
        with granular.ShardedDatasetWriter(root / 'train', spec, None) as w:
            w.append({'foo': b''})
        granular.commit(root / 'train')
        with granular.ShardedDatasetReader(root / 'train', None) as r:
            assert len(r) == 1
        with pytest.raises(PermissionError):
            granular.ShardedDatasetWriter(root / 'train', spec, None)
        with pytest.raises(PermissionError):
            granular.DatasetWriter(root / 'train' / '000000', spec, None)
        granular.DatasetWriter(root / 'test', spec, None).close()
