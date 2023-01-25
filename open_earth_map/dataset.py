import math
import numpy as np
import torch
import rasterio
from PIL import Image
from . import transforms


def load_multiband(path: str):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path: str):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class OpenEarthMapDataset(torch.utils.data.Dataset):

    """
    OpenEarthMap dataset

    Args:
        fn_list (str): List containing image names
        classes (int): list of of class-code
        augm (Classes): transfromation pipeline (e.g. Rotate, Crop, etc.)
    """

    def __init__(self, img_list: list, n_classes: int = 9, testing=False, augm=None):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("images", "labels") for f in self.fn_imgs]
        self.augm = augm
        self.testing = testing
        self.classes = np.arange(n_classes).tolist()
        self.to_tensor = transforms.ToTensor(classes=self.classes)

        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = Image.fromarray(self.load_multiband(self.fn_imgs[idx]))

        if not self.testing:
            msk = Image.fromarray(self.load_grayscale(self.fn_msks[idx]))
        else:
            msk = Image.fromarray(np.zeros(img.size[:2], dtype="uint8"))

        if self.augm is not None:
            data = self.augm({"image": img, "mask": msk})
        else:
            h, w = msk.size
            power_h = math.ceil(np.log2(h) / np.log2(2))
            power_w = math.ceil(np.log2(w) / np.log2(2))
            if 2**power_h != h or 2**power_w != w:
                img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
                msk = msk.resize((2**power_w, 2**power_h), resample=Image.NEAREST)
            data = {"image": img, "mask": msk}

        data = self.to_tensor(
            {
                "image": np.array(data["image"], dtype="uint8"),
                "mask": np.array(data["mask"], dtype="uint8"),
            }
        )
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)
