import json
import pathlib


META = 'meta.json'


def commit(directory):
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    meta = readmeta(directory)
    assert meta, directory
    meta['done'] = True
    (directory / META).write_bytes(json.dumps(meta).encode('utf-8'))


def writemeta(directory, spec):
    if committed(directory):
        raise PermissionError(
            f'Cannot write to {directory}: dataset is committed. '
            'Write a new version instead.'
        )
    meta = readmeta(directory)
    if meta:
        assert spec == meta['spec'], (spec, meta['spec'])
        return
    directory.mkdir(exist_ok=True, parents=True)
    meta = {'spec': spec, 'done': False}
    (directory / META).write_bytes(json.dumps(meta).encode('utf-8'))


def committed(directory):
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    metas = [readmeta(d) for d in (directory, directory.parent)]
    return any(meta and meta['done'] for meta in metas)


def readmeta(directory):
    filename = directory / META
    if filename.exists():
        return json.loads(filename.read_bytes())
    legacy = directory / 'spec.json'
    if legacy.exists():
        return {'spec': json.loads(legacy.read_bytes()), 'done': False}
    return None


class Closing:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        # assert not self.closed
        return self

    def __exit__(self, *e):
        try:
            self.close()
        except Exception:
            if not e[0]:
                raise
        self.closed = True
