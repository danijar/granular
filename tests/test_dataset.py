import json
import pathlib

import cloudpickle
import granular
import pytest


class TestDataset:
    def test_writer(self, tmpdir):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int', 'baz': 'utf8'}
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            for i in range(10):
                index = writer.append(
                    {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                )
                assert index == i
                assert len(writer) == i + 1
            assert len(writer) == 10
            assert writer.size > 0
        assert set(x.name for x in directory.glob('*')) == {
            'spec.json',
            'foo.bag',
            'bar.bag',
            'baz.bag',
        }
        spec2 = json.loads((directory / 'spec.json').read_bytes())
        assert list(spec2.keys()) == sorted(spec2.keys())
        assert set(spec.keys()) == set(spec2.keys())
        assert spec == spec2

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize(
        'cache_keys',
        (
            [],
            ['foo'],
            ['foo', 'bar'],
            ['foo', 'bar', 'baz'],
        ),
    )
    @pytest.mark.parametrize('parallel', (True, False))
    def test_roundtrip(self, tmpdir, cache_index, cache_keys, parallel):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        datapoints = []
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
                datapoints.append(datapoint)
            size = writer.size
        with granular.DatasetReader(
            directory, granular.decoders, cache_index, cache_keys, parallel
        ) as reader:
            assert len(reader) == 10
            assert reader.size == size
            for i in range(10):
                datapoint = reader[i]
                assert datapoint == datapoints[i]

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize('cache_keys', ([], ['foo'], ['foo', 'bar']))
    @pytest.mark.parametrize('parallel', (True, False))
    def test_masking(self, tmpdir, cache_index, cache_keys, parallel):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int', 'baz': 'utf8'}
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
        with granular.DatasetReader(
            directory, granular.decoders, cache_index, cache_keys, parallel
        ) as reader:
            assert reader[3, ()] == {}
            assert reader[3, ('foo',)] == {'foo': 'hello world'}
            assert reader[3, ('foo', 'baz')] == {
                'baz': 'word3',
                'foo': 'hello world',
            }

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize('cache_keys', ([], ['foo'], ['foo', 'bar']))
    @pytest.mark.parametrize('parallel', (True, False))
    def test_slicing(self, tmpdir, cache_index, cache_keys, parallel):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'foo': 'utf8', 'bar': 'int'}
        datapoints = [{'foo': 'hello world', 'bar': i} for i in range(10)]
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            [writer.append(x) for x in datapoints]
        with granular.DatasetReader(
            directory, granular.decoders, cache_index, cache_keys, parallel
        ) as reader:
            stacked = {k: [x[k] for x in datapoints] for k in datapoints[0]}
            assert reader[0:10] == stacked
            assert reader[2:5] == {k: v[2:5] for k, v in stacked.items()}
            assert reader[2:3, ('foo',)] == {'foo': ['hello world']}
            assert reader[2:4, ('foo',)] == {'foo': ['hello world'] * 2}
            assert reader[2:4, ('bar',)] == {
                'bar': [datapoints[2]['bar'], datapoints[3]['bar']]
            }

    @pytest.mark.parametrize('cache_index', (True, False))
    @pytest.mark.parametrize('cache_keys', ([], ['foo'], ['foo', 'baz']))
    @pytest.mark.parametrize('parallel', (True, False))
    def test_pickle(self, tmpdir, cache_index, cache_keys, parallel):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        datapoints = []
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
                datapoints.append(datapoint)
            size = writer.size
        reader = granular.DatasetReader(
            directory, granular.decoders, cache_index, cache_keys, parallel
        )
        reader = cloudpickle.loads(cloudpickle.dumps(reader))
        reader = cloudpickle.loads(cloudpickle.dumps(reader))
        with reader:
            assert len(reader) == 10
            assert reader.size == size
            for i in range(10):
                datapoint = reader[i]
                assert datapoint == datapoints[i]

    def test_parallel_exit(self, tmpdir):
        directory = pathlib.Path(tmpdir) / 'dataset'
        spec = {'bar': 'int', 'baz': 'utf8', 'foo': 'utf8'}
        datapoints = []
        with granular.DatasetWriter(
            directory, spec, granular.encoders
        ) as writer:
            for i in range(10):
                datapoint = {'foo': 'hello world', 'bar': i, 'baz': f'word{i}'}
                writer.append(datapoint)
                datapoints.append(datapoint)
            size = writer.size
        reader = granular.DatasetReader(
            directory, granular.decoders, parallel=True
        )
        assert len(reader) == 10
        assert reader.size == size
        for i in range(10):
            datapoint = reader[i]
            assert datapoint == datapoints[i]
        # Intentionally omitted reader.close().
