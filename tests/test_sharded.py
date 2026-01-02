import pathlib

import granular
import numpy as np
import pytest


class TestSharded:
    @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
    def test_writer(self, tmpdir, shardlen):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            assert writer.spec == spec
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
            if shardlen:
                assert writer.shards == int(np.ceil(10 / shardlen))
            else:
                assert writer.shards == 1
        assert set(x.name for x in directory.glob('*')) == {
            f'{i:06}' for i in range(writer.shards)
        }
        for folder in directory.glob('*'):
            assert set(x.name for x in folder.glob('*')) == {
                'spec.json',
                'bar.bag',
                'baz.bag',
                'foo.bag',
                'bar.idx',
                'baz.idx',
                'foo.idx',
            }

    @pytest.mark.parametrize('shardlen', (None, 1, 5))
    def test_length(self, tmpdir, shardlen):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
            if shardlen:
                assert writer.shards == int(np.ceil(10 / shardlen))
            else:
                assert writer.shards == 1

    @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
    @pytest.mark.parametrize('parallel', (True, False))
    def test_roundtrip(self, tmpdir, shardlen, parallel):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        datapoints = []
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            assert writer.spec == spec
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
                datapoints.append(datapoint)
            shards = writer.shards
            size = writer.size
        with granular.ShardedDatasetReader(
            directory, granular.decoders, parallel=parallel
        ) as reader:
            assert reader.spec == spec
            assert reader.shards == shards
            assert reader.size == size
            assert len(reader) == 10
            for i in range(10):
                assert reader[i] == datapoints[i]

    @pytest.mark.parametrize('shardlen', (1, 3, 10, 15))
    @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
    def test_worker_writing(self, tmpdir, shardlen, nworkers):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8'}
        datapoints = [{'bar': i, 'baz': f'hello{i}'} for i in range(10)]
        shards = 0
        size = 0
        for worker in range(nworkers):
            with granular.ShardedDatasetWriter(
                directory,
                spec,
                granular.encoders,
                shardlen,
                shardstart=worker,
                shardstep=nworkers,
            ) as writer:
                for i in range(worker, 10, nworkers):
                    writer.append(datapoints[i])
                shards += writer.shards
                size += writer.size
        with granular.ShardedDatasetReader(
            directory, granular.decoders
        ) as reader:
            assert reader.shards == shards
            assert reader.size == size
            assert len(reader) == 10
            received = [reader[i] for i in range(10)]
            received = sorted(received, key=lambda x: x['bar'])
            assert datapoints == received

    @pytest.mark.parametrize('shardlen', (1, 3, 10, 15))
    @pytest.mark.parametrize('nworkers', (1, 2, 3, 10))
    def test_worker_roundtrip(self, tmpdir, shardlen, nworkers):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8'}
        datapoints = [{'bar': i, 'baz': f'hello{i}'} for i in range(10)]
        shards = 0
        size = 0
        for worker in range(nworkers):
            with granular.ShardedDatasetWriter(
                directory,
                spec,
                granular.encoders,
                shardlen,
                shardstart=worker,
                shardstep=nworkers,
            ) as writer:
                for i in range(worker, 10, nworkers):
                    writer.append(datapoints[i])
                shards += writer.shards
                size += writer.size
        received = []
        for worker in range(nworkers):
            with granular.ShardedDatasetReader(
                directory,
                granular.decoders,
                shardstart=worker,
                shardstep=nworkers,
            ) as reader:
                received += [reader[i] for i in range(len(reader))]
        received = sorted(received, key=lambda x: x['bar'])
        assert datapoints == received

    @pytest.mark.parametrize('shardlen', (None, 1, 3, 10, 15))
    def test_slicing(self, tmpdir, shardlen):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        datapoints = [{'foo': 'hello world', 'bar': i} for i in range(10)]
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            [writer.append(x) for x in datapoints]
        with granular.ShardedDatasetReader(
            directory, granular.decoders
        ) as reader:
            stacked = {k: [x[k] for x in datapoints] for k in datapoints[0]}
            assert reader[0:10] == stacked
            assert reader[2:5] == {k: v[2:5] for k, v in stacked.items()}
            assert reader[2:3, ('foo',)] == {'foo': ['hello world']}
            assert reader[2:4, ('foo',)] == {'foo': ['hello world'] * 2}
            assert reader[2:4, ('bar',)] == {
                'bar': [datapoints[2]['bar'], datapoints[3]['bar']]
            }

    @pytest.mark.parametrize('shardlen', (5, 10, None))
    def test_resume_after_close(self, tmpdir, shardlen):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            for i in range(5):
                writer.append({'foo': f'value{i}', 'bar': i})
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            for i in range(5, 10):
                writer.append({'foo': f'value{i}', 'bar': i})
        with granular.ShardedDatasetReader(
            directory, granular.decoders
        ) as reader:
            assert len(reader) == 10
            for i in range(10):
                assert reader[i] == {'foo': f'value{i}', 'bar': i}

    @pytest.mark.parametrize('extra_records', [1, 2, 3])
    def test_preemption_partial_write(self, tmpdir, extra_records):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen=None
        ) as writer:
            for i in range(5):
                writer.append({'foo': f'value{i}', 'bar': i})
            current_shard = writer.shardnum
        shard_dir = directory / f'{current_shard:06}'
        with granular.BagWriter(shard_dir / 'foo.bag') as bag_writer:
            for i in range(5, 5 + extra_records):
                bag_writer.append(granular.encoders['utf8'](f'value{i}'))
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen=None
        ) as writer:
            for i in range(5, 5 + extra_records + 2):
                writer.append({'foo': f'value{i}', 'bar': i})
        with granular.ShardedDatasetReader(
            directory, granular.decoders
        ) as reader:
            assert len(reader) == 5 + extra_records + 2
            for i in range(5 + extra_records + 2):
                assert reader[i] == {'foo': f'value{i}', 'bar': i}

    def test_preemption_mismatch_raises(self, tmpdir):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen=None
        ) as writer:
            for i in range(5):
                writer.append({'foo': f'value{i}', 'bar': i})
            current_shard = writer.shardnum
        shard_dir = directory / f'{current_shard:06}'
        with granular.BagWriter(shard_dir / 'foo.bag') as bag_writer:
            bag_writer.append(granular.encoders['utf8']('WRONG_VALUE'))
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen=None
        ) as writer:
            with pytest.raises(ValueError, match='Record mismatch'):
                writer.append({'foo': 'value5', 'bar': 5})

    @pytest.mark.parametrize('extra_records', [1, 2])
    def test_preemption_partial_shard(self, tmpdir, extra_records):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        shardlen = 5
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            for i in range(3):
                writer.append({'foo': f'value{i}', 'bar': i})
        shard_dir = directory / '000000'
        with granular.BagWriter(shard_dir / 'foo.bag') as bag_writer:
            for i in range(3, 3 + extra_records):
                bag_writer.append(granular.encoders['utf8'](f'value{i}'))
        with granular.ShardedDatasetWriter(
            directory, spec, granular.encoders, shardlen
        ) as writer:
            for i in range(3, 3 + extra_records + 4):
                writer.append({'foo': f'value{i}', 'bar': i})
        with granular.ShardedDatasetReader(
            directory, granular.decoders
        ) as reader:
            total = 3 + extra_records + 4
            assert len(reader) == total
            for i in range(total):
                assert reader[i] == {'foo': f'value{i}', 'bar': i}
