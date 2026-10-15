import pathlib


DONE = 'DONE'


def commit(directory):
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    filename = directory / DONE
    if not filename.exists():
        filename.write_bytes(b'')


def committed(directory):
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    return any((d / DONE).exists() for d in (directory, directory.parent))


def check_writable(directory):
    if committed(directory):
        raise PermissionError(
            f'Cannot write to {directory}: dataset is committed. '
            'Write a new version instead.'
        )


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
