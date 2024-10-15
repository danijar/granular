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
