import io

import msgpack


class Integer:

  @staticmethod
  def encode(value, size=None, endian='little'):
    import numpy as np
    if size is None:
      size = np.ceil(np.log2(1 + value) / 8)
    return value.to_bytes(int(size), endian)

  @staticmethod
  def decode(buffer, size=None, endian='little'):
    return int.from_bytes(buffer, endian)


class Array:

  @staticmethod
  def encode(value, dtype, *shape):
    import numpy as np
    assert value.dtype == np.dtype(dtype)
    assert value.shape == tuple(int(x) for x in shape)
    return value.tobytes()

  @staticmethod
  def decode(buffer, dtype, *shape):
    import numpy as np
    shape = tuple(int(x) for x in shape)
    return np.frombuffer(buffer, dtype).reshape(shape)


class JPEG:

  @staticmethod
  def encode(value):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(value).save(stream, format='JPEG')
    return stream.getvalue()

  @staticmethod
  def decode(buffer):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buffer)))


encoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'msgpack': msgpack.packb,
    'int': Integer.encode,
    'array': Array.encode,
    'jpeg': JPEG.encode,
}

decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'msgpack': msgpack.unpackb,
    'int': Integer.decode,
    'array': Array.decode,
    'jpeg': JPEG.decode,
}
