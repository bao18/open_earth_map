import random
import numbers
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_tensor(np.stack(msks, axis=-1))
        sample["image"] = TF.to_tensor(sample["image"])
        return sample


class Rotate:
    def __init__(self, degrees=(-180, 180)):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(*self.degrees)

        img = TF.rotate(sample["image"], angle, InterpolationMode.BICUBIC)
        msk = TF.rotate(sample["mask"], angle, InterpolationMode.NEAREST)
        return {"image": img, "mask": msk}


class Crop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["mask"].size

        if h > self.size[0] and w > self.size[1]:
            i = random.randrange(0, h - self.size[0])
            j = random.randrange(0, w - self.size[1])
            img = TF.crop(sample["image"], i, j, *self.size)
            msk = TF.crop(sample["mask"], i, j, *self.size)
        else:
            img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
            msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
        msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}
