import numpy as np
from torchvision import transforms

from ImageNet.data_transform import Compose, LabelGuard, LabelToArray


# Datasets path
train_path = r'E:\ILSVRC\Data\CLS-LOC\train'
dev_path = r'E:\ILSVRC\Data\CLS-LOC\dev'

# Annotations
train_data = r'annotations\train_labels.txt'
dev_data = r'annotations\dev_labels.txt'


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Data Transformations
default_transform = Compose([
    LabelGuard(transforms.Resize(256)),
    LabelGuard(transforms.RandomCrop(224)),
    LabelGuard(transforms.ToTensor()),
    LabelGuard(transforms.Normalize(mean, std)),
    LabelToArray()
])

# The inverse transform only takes an image as input
inverse_default_transform = transforms.Compose([
    transforms.Normalize(-mean / std, 1 / std),
    transforms.ToPILImage()
])
