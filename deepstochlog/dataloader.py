import random
from math import ceil
from typing import Sequence, List


class DataLoader(object):
    def __init__(
        self,
        dataset: Sequence[any],
        batch_size: int,
        shuffle: bool = True,
        max_size: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if max_size is not None:
            self.length = min(self.length, max_size)
        else:
            self.length = len(dataset)

        self.dataset = list(self.dataset[: self.length])

        # Set up the iterator
        self.idx_iterator = None
        self._create_new_iterator()

    def _create_new_iterator(self):
        idxs = list(range(self.length))
        if self.shuffle:
            random.shuffle(idxs)
        self.idx_iterator = iter(idxs)

    def __next__(self) -> List:
        if self.idx_iterator is None:
            self._create_new_iterator()
            raise StopIteration
        batch = list()
        try:
            for _ in range(self.batch_size):
                batch.append(self.dataset[next(self.idx_iterator)])
        except StopIteration:
            if len(batch) == 0:
                self._create_new_iterator()
                raise StopIteration
            else:
                self.idx_iterator = None
        return batch

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.length / self.batch_size))

    def __repr__(self):
        return "DataLoader: " + str(self.dataset[0])
