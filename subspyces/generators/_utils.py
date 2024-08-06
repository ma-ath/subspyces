from collections.abc import Iterable


def batched_iterable(iterable: Iterable, batch_size: int):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]
