import numpy as np


class CrossValidation:

    def __init__(self, sample_count, fold_count):
        sample_ids = np.array(range(sample_count))
        np.random.shuffle(sample_ids)
        self.partitions = {}
        partition_size = int( np.ceil(float(sample_count) / float(fold_count)))
        for i in range(fold_count):
            self.partitions[i] = sample_ids[i * partition_size:min(sample_count, (i + 1) * partition_size)]
        self.currPartition = 0
        self.foldCount = fold_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.currPartition < self.foldCount:
            cp = self.currPartition
            self.currPartition += 1
            return self.partitions[cp]
        else:
            raise StopIteration()
