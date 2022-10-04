#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Contact: bruno.adriano@riken.jp

import numpy as np
import torch
import rasterio
from . import transforms


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class OpenEarthMapDataset(torch.utils.data.Dataset):

    """
    OpenEarthMap dataset
    Geoinformatics Unit, RIKEN AIP

    Args:
        fn_list (str): List containing images paths
        classes (int): list of of class-code
        img_size (int): image size
        augm (albumentations): transfromation pipeline (e.g. flip, cut, etc.)
    """

    def __init__(
        self, img_list, classes, img_size=512, augm=None, mu=None, sig=None,
    ):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("/images/", "/labels/") for f in self.fn_imgs]
        self.augm = augm
        self.to_tensor = (
            transforms.ToTensor(classes=classes)
            if mu is None
            else transforms.ToTensorNorm(classes=classes, mu=mu, sig=sig)
        )
        self.size = img_size
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fn_imgs[idx])
        msk = self.load_grayscale(self.fn_msks[idx])

        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)
