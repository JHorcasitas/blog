from torchvision import transforms

from ImageNet.data_transform import Compose, LabelGuard, LabelToArray


# Datasets path
train_path = r'E:\ILSVRC\Data\CLS-LOC\train'
dev_path = r'E:\ILSVRC\Data\CLS-LOC\dev'

# Annotations
train_data = r'C:\Users\artur\Desktop\artur\blog\annotations\train_labels.txt'
dev_data = r'C:\Users\artur\Desktop\artur\blog\annotations\dev_labels.txt'


# Data Transformations
default_transform = Compose([
    LabelGuard(transforms.Resize(256)),
    LabelGuard(transforms.RandomCrop(224)),
    LabelGuard(transforms.ToTensor()),
    LabelGuard(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])),
    LabelToArray()
])


inverse_default_transform = None
