import torch


class LabelGuard:
    """Allows torchvision transformation to be called with an additional
    argument representing the image label."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        return self.transform(img), label


class LabelToTensor:
    def __call__(self, img, label):
        return img, torch.tensor(int(label), dtype=torch.long)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label
