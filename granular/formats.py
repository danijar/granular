import io
from functools import partial as bind

import msgpack


def encode_int(value, size=None, endian='little'):
  import numpy as np
  if size is None:
    size = np.ceil(np.log2(1 + value) / 8)
  return value.to_bytes(int(size), endian)


def decode_int(buffer, size=None, endian='little'):
  return int.from_bytes(buffer, endian)


def encode_array(value):
  assert value.data.c_contiguous
  header = msgpack.packb((value.dtype.str, value.shape))
  hsize = len(header).to_bytes(4, 'little')
  return hsize + header + value.data


def decode_array(buffer):
  import numpy as np
  hsize = int.from_bytes(buffer[:4], 'little')
  dtype, shape = msgpack.unpackb(buffer[4: 4 + hsize])
  return np.frombuffer(buffer[4 + hsize:], dtype).reshape(shape)


def encode_image(value, quality=100, format='jpg'):
  format = ('jpeg' if format == 'jpg' else format).upper()
  from PIL import Image
  stream = io.BytesIO()
  Image.fromarray(value).save(stream, format=format)
  return stream.getvalue()


def decode_image(buffer, *args):
  import numpy as np
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
    'int': encode_int,
    'array': encode_array,
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
    'jpg': decode_image,
    'png': decode_image,
    'mp4': decode_mp4,
}
