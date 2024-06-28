import functools
import io

import msgpack


def int_encode(value, size=None, endian='little'):
  import numpy as np
  if size is None:
    size = np.ceil(np.log2(1 + value) / 8)
  return value.to_bytes(int(size), endian)


def int_decode(buffer, size=None, endian='little'):
  return int.from_bytes(buffer, endian)


def array_encode(value, dtype, *shape):
  import numpy as np
  assert value.dtype == np.dtype(dtype)
  assert value.shape == tuple(int(x) for x in shape)
  return value.tobytes()


def array_decode(buffer, dtype, *shape):
  import numpy as np
  shape = tuple(int(x) for x in shape)
  return np.frombuffer(buffer, dtype).reshape(shape)


def image_encode(value, quality=None, format='jpeg'):
  from PIL import Image
  stream = io.BytesIO()
  Image.fromarray(value).save(stream, format=format.upper())
  return stream.getvalue()


def image_decode(buffer, *args):
  import numpy as np
  from PIL import Image
  return np.asarray(Image.open(io.BytesIO(buffer)))


encoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'msgpack': msgpack.packb,
    'int': int_encode,
    'array': array_encode,
    'jpeg': functools.partial(image_encode, format='jpeg'),
    'png': functools.partial(image_encode, format='png'),
}

decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'msgpack': msgpack.unpackb,
    'int': int_decode,
    'array': array_decode,
    'jpeg': image_decode,
    'png': image_decode,
}
