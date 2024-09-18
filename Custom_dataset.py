from torch.utils.data import Subset


class CustomDataset(Subset):
    def __init__(self, subset, offset=0):
        super(CustomDataset, self).__init__(subset.dataset, subset.indices)
        self.offset = offset

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label + self.offset