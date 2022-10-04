#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Contact: bruno.adriano@riken.jp

import numpy as np
import cv2
import torch
import albumentations as A
import torchvision.transforms.functional as TF


class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_tensor(np.stack(msks, axis=-1))
        sample["image"] = TF.to_tensor(sample["image"])
        return sample


class ToTensorNorm:
    def __init__(self, classes, mu, sig):
        self.classes = classes
        self.mu = mu
        self.sig = sig

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        msk = TF.to_tensor(np.stack(msks, axis=-1))

        imgs = torch.from_numpy(np.moveaxis(sample["image"], -1, 0))
        img = TF.normalize(imgs.float(), self.mu, self.sig)
        return {"image": img, "mask": msk}


def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, interpolation=cv2.INTER_NEAREST, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm(sample, size=512):
    augms = A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])


def train_augm_full(sample, size=512):
    augms = A.Compose(
        [
            A.ShiftScaleRotate(
                scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7,
            ),
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
            A.Flip(p=0.5),
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
            A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
            # color transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(70, 130), p=1),
                    A.ChannelShuffle(p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=30,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0,
                    ),
                ],
                p=0.8,
            ),
            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=1),
                    A.OpticalDistortion(p=1),
                    A.GridDistortion(p=1),
                    A.Perspective(p=1),
                ],
                p=0.2,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.2,
            ),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])


def train_augm_color(sample, size=512):
    augms = A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1),
            A.RandomCrop(size, size, p=1.0),
            # color transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(70, 130), p=1),
                    A.ChannelShuffle(p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=30,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0,
                    ),
                ],
                p=0.8,
            ),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])


def train_augm_distortion(sample, size=512):
    augms = A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1),
            A.RandomCrop(size, size, p=1.0),
            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=1),
                    A.OpticalDistortion(p=1),
                    A.GridDistortion(p=1),
                    A.Perspective(p=1),
                ],
                p=0.2,
            ),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])


def train_augm_noise(sample, size=512):
    augms = A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1),
            A.RandomCrop(size, size, p=1.0),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.2,
            ),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])


def train_augm_other(sample, size=512):
    augms = A.Compose(
        [
            A.ShiftScaleRotate(
                scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7,
            ),
            A.PadIfNeeded(size, size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(size, size, p=1.0),
            A.Flip(p=0.5),
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
            A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1,),
        ]
    )
    return augms(image=sample["image"], mask=sample["mask"])
