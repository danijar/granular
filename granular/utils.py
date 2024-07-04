class Closing:

  def __init__(self):
    self.closed = False

  def __enter__(self):
    # assert not self.closed
    return self

  def __exit__(self, *e):
    self.close()
    self.closed = True
