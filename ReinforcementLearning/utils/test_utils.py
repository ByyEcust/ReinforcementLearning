from utils.utils_all import Buffer, RollAverage


def test_buffer():
    buffer = Buffer(10)
    for _ in range(5):
        buffer.append([_, _**2, _**3])
    num2sample = 6
    sample_batch = buffer.sample_batch(num2sample)
    assert len(sample_batch) == num2sample, "The sampled size is %d, but the expected one is %d.".\
        format(len(sample_batch), num2sample)

    for sample in sample_batch:
        assert sample is not None, "The sampled instance is with None."


def test_roll_average():
    roll_average = RollAverage(3)
    res = []
    for i in range(5):
        res.append(roll_average.update(i))
    assert res == [0, 0.5, 1, 2, 3]

