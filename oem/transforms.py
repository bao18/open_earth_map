#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Contact: bruno.adriano@riken.jp

import random
import numbers
import numpy as np
import PIL
import torchvision.transforms.functional as TF


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_tensor(np.stack(msks, axis=-1))
        sample["image"] = TF.to_tensor(sample["image"])
        return sample


class RandomRotate:
    def __init__(self, degrees=(-180, 180)):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(*self.degrees)

        img = TF.rotate(sample["image"], angle, interpolation=PIL.Image.BICUBIC)
        msk = TF.rotate(sample["mask"], angle, interpolation=PIL.Image.NEAREST)
        return {"image": img, "mask": msk}


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["mask"].size
        i = random.randrange(0, h - self.size[0])
        j = random.randrange(0, w - self.size[1])
        return {
            "image": TF.crop(sample["image"], i, j, *self.size),
            "mask": TF.crop(sample["mask"], i, j, *self.size),
        }


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        img = TF.resize(sample["image"], self.size, interpolation=PIL.Image.BICUBIC)
        msk = TF.resize(sample["mask"], self.size, interpolation=PIL.Image.NEAREST)

        return {"image": img, "mask": msk}
