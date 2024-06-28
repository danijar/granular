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


def mp4_encode(array, fps=15):
  import av
  T, H, W = array.shape[:3]
  fp = io.BytesIO()
  output = av.open(fp, mode='w', format='mp4')
  stream = output.add_stream('mpeg4', rate=float(fps))
  stream.width = W
  stream.height = H
  stream.pix_fmt = 'yuv420p'
  for t in range(T):
    frame = av.VideoFrame.from_ndarray(array[t], format='rgb24')
    frame.pts = t
    output.mux(stream.encode(frame))
  output.mux(stream.encode(None))
  output.close()
  return fp.getvalue()


def mp4_decode(buffer, *args):
  import numpy as np
  import av
  container = av.open(io.BytesIO(buffer))
  stream = container.streams.video[0]
  T, H, W = stream.frames, stream.height, stream.width
  array = np.empty((T, H, W, 3), dtype=np.uint8)
  for t, frame in enumerate(container.decode(video=0)):
    array[t] = frame.to_ndarray(format='rgb24')
  container.close()
  return array


encoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.encode('utf-8'),
    'msgpack': msgpack.packb,
    'int': int_encode,
    'array': array_encode,
    'jpeg': functools.partial(image_encode, format='jpeg'),
    'png': functools.partial(image_encode, format='png'),
    'mp4': mp4_encode,
}

decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'msgpack': msgpack.unpackb,
    'int': int_decode,
    'array': array_decode,
    'jpeg': image_decode,
    'png': image_decode,
    'mp4': mp4_decode,
}
