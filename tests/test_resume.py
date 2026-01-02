import pathlib

import granular
import pytest


class TestResume:
    def test_resume_after_close(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with granular.BagWriter(bag) as writer:
            writer.append(b'record1')
            writer.append(b'record2')
        with granular.BagWriter(bag) as writer:
            assert len(writer) == 2
            writer.append(b'record3')
            assert len(writer) == 3
        with granular.BagReader(bag) as reader:
            assert len(reader) == 3
            assert reader[0] == b'record1'
            assert reader[1] == b'record2'
            assert reader[2] == b'record3'

    def test_resume_with_partial_record(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with granular.BagWriter(bag) as writer:
            writer.append(b'record1')
            writer.append(b'record2')
        with bag.open('ab') as f:
            f.write(b'rec')
        with granular.BagWriter(bag) as writer:
            assert len(writer) == 2
            writer.append(b'record3')
        with granular.BagReader(bag) as reader:
            assert len(reader) == 3
            assert reader[0] == b'record1'
            assert reader[1] == b'record2'
            assert reader[2] == b'record3'
        assert bag.stat().st_size == len(b'record1record2record3')

    def test_resume_with_full_record(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with granular.BagWriter(bag) as writer:
            writer.append(b'record1')
        with bag.open('ab') as f:
            f.write(b'record2')
        with granular.BagWriter(bag) as writer:
            assert len(writer) == 1
            writer.append(b'record2')
        with granular.BagReader(bag) as reader:
            assert len(reader) == 2
            assert reader[1] == b'record2'
        assert bag.stat().st_size == len(b'record1record2')

    def test_resume_partial_mismatch_raises(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with granular.BagWriter(bag) as writer:
            writer.append(b'record1')
        with bag.open('ab') as f:
            f.write(b'other')
        writer = granular.BagWriter(bag)
        assert len(writer) == 1
        writer.append(b'record2', flush=False)
        with pytest.raises(ValueError, match='does not match'):
            writer.flush()

    def test_resume_empty(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with granular.BagWriter(bag) as writer:
            pass
        with granular.BagWriter(bag) as writer:
            assert len(writer) == 0
            writer.append(b'first')
        with granular.BagReader(bag) as reader:
            assert len(reader) == 1

    def test_resume_bag_exists_idx_missing_raises(self, tmpdir):
        bag = pathlib.Path(tmpdir) / 'file.bag'
        with bag.open('wb') as f:
            f.write(b'garbage')
        writer = granular.BagWriter(bag)
        assert len(writer) == 0
        writer.append(b'first', flush=False)
        with pytest.raises(ValueError, match='does not match'):
            writer.flush()
