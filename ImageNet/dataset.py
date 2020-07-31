from os.path import join as join_path
from typing import Optional, Tuple, List, Callable
from collections import namedtuple

from PIL import Image
from numpy import ndarray

from torch.utils.data import Dataset

from ImageNet import config


def read_data(kind: str) -> List[namedtuple]:
    records = []
    fpath = config.train_data_path if kind == 'train' else config.dev_data_path
    Record = namedtuple('Record', ['fname', 'id', 'label', 'class_name'])
    with open(fpath, 'rt') as f:
        for line in f:
            line = line.strip('\n').split()
            r = Record(fname=line[0],
                       id=line[1],
                       label=line[2],
                       class_name=line[3])
            records.append(r)
    return records


class ImageNetDataset(Dataset):
    """Load images from ImageNet dataset.

    :param kind: Either 'train' or 'dev'.
    :param transform: Transformation to apply to the sample.
    """
    def __init__(self,
                 kind: str,
                 transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.img_paths, self.labels = [], []
        for record in read_data(kind):
            self.img_paths.append(join_path(config.train_path, record.fname))
            self.labels.append(record.label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label
