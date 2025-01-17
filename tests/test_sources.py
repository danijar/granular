import granular


class TestSources:

  def test_transform(self):
    data = list(range(10))
    source = granular.sources.Transform(data, (lambda x, seed: 2 * x))
    output = [source(step) for step in range(10)]
    assert output == [2 * x for x in data]

  def test_sample(self):
    data = list(range(10))
    source = granular.sources.Sample(data)
    output = [source(step) for step in range(10)]
    assert output != data
    assert sorted(output) != data
    assert set(output) <= set(data)

  def test_truncate(self):
    data = list(range(10))
    source = granular.sources.Truncate(data, 5)
    output = [source(step) for step in range(10)]
    assert output == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

  def test_epochs_ordered(self):
    data = list(range(10))
    source = granular.sources.Epochs(data, shuffle=False)
    output = [source(step) for step in range(20)]
    assert output[:10] == data
    assert output[10:] == data

  def test_epochs_shuffle(self):
    data = list(range(10))
    source = granular.sources.Epochs(data, shuffle=True)
    output = [source(step) for step in range(20)]
    assert output[:10] != data
    assert output[10:] != data
    assert sorted(output[:10]) == data
    assert sorted(output[10:]) == data

  def test_interleave(self):
    data1 = list(range(10))
    data2 = list('abcdefghij')
    source = granular.sources.Interleave([data1, data2])
    output = [source(step) for step in range(20)]
    assert output[::2] == data1
    assert output[1::2] == data2

  def test_mix(self, thres=0.2):
    data1 = list(range(10))
    data2 = list('abcdefghij')
    source1 = granular.sources.Sample(data1)
    source2 = granular.sources.Sample(data2)
    source = granular.sources.Mix([source1, source2], [0.2, 0.8])
    output = [source(step) for step in range(1000)]
    output1 = [x for x in output if x in data1]
    output2 = [x for x in output if x in data2]
    assert set(output) == set(data1 + data2)
    frac1 = len(output1) / len(output)
    frac2 = len(output2) / len(output)
    assert (1 - thres) * 0.2 <= frac1 <= (1 + thres) * 0.2
    assert (1 - thres) * 0.8 <= frac2 <= (1 + thres) * 0.8
