import io
from functools import partial as bind

import msgpack
import numpy as np


def encode_int(value, size=None, endian='little'):
  if size is None:
    size = np.ceil(np.log2(1 + value) / 8)
  return value.to_bytes(int(size), endian)


def decode_int(buffer, size=None, endian='little'):
  assert len(buffer) <= 16, len(buffer)
  return int.from_bytes(buffer, endian)


def encode_array(value):
  assert value.data.c_contiguous
  return msgpack.packb((value.dtype.str, value.shape, value.data))


def decode_array(buffer):
  dtype, shape, data = msgpack.unpackb(buffer)
  return np.frombuffer(data, dtype).reshape(shape)


def encode_tree(value):
  def fn(xs):
    if isinstance(xs, (list, tuple)):
      return [fn(x) for x in xs]
    elif isinstance(xs, dict):
      return {k: fn(v) for k, v in xs.items()}
    elif isinstance(xs, np.ndarray):
      assert xs.data.c_contiguous
      return ('_', xs.dtype.str, xs.shape, xs.data)
    else:
      return xs
  return msgpack.packb(fn(value))


def decode_tree(buffer):
  def fn(xs):
    if isinstance(xs, list) and len(xs) == 4 and xs[0] == '_':
      _, dtype, shape, data = xs
      return np.frombuffer(data, dtype).reshape(shape)
    elif isinstance(xs, (list, tuple)):
      return [fn(x) for x in xs]
    elif isinstance(xs, dict):
      return {k: fn(v) for k, v in xs.items()}
    else:
      return xs
  return fn(msgpack.unpackb(buffer))


def encode_image(value, quality=100, format='jpg'):
  format = ('jpeg' if format == 'jpg' else format).upper()
  from PIL import Image
  stream = io.BytesIO()
  Image.fromarray(value).save(stream, format=format)
  return stream.getvalue()


def decode_image(buffer, *args):
  from PIL import Image
  return np.asarray(Image.open(io.BytesIO(buffer)))


def encode_mp4(array, fps=20):
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


def decode_mp4(buffer, *args):
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
    'int': encode_int,
    'array': encode_array,
    'tree': encode_tree,
    'jpg': bind(encode_image, format='jpg'),
    'png': bind(encode_image, format='png'),
    'mp4': encode_mp4,
}


decoders = {
    'bytes': lambda x: x,
    'utf8': lambda x: x.decode('utf-8'),
    'msgpack': msgpack.unpackb,
    'int': decode_int,
    'array': decode_array,
    'tree': decode_tree,
    'jpg': decode_image,
    'png': decode_image,
    'mp4': decode_mp4,
}
