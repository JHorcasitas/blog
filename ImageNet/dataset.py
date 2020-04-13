# import yaml
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class DatasetFactory:

    @staticmethod
    def get_dataset(split: str) -> Dataset:
        if split == 'train':
            return ImageFolder(root=)
        elif split == 'val':
            pass
        else:
            raise ValueError()


class ValidationtDataset:
    pass
