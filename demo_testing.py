import os
import time
import warnings
import numpy as np
import torch
import oem
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    start = time.time()

    # Path to the OpenEarthMap directory
    OEM_DATA_DIR = "/home/bruno/riken_openearthmap/LULC-RIKEN/integrated"

    # Path to test file list
    TEST_LIST = "data/test_mini.txt"
    N_CLASSES = 9
    DEVICE = "cuda"
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_paths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    test_fns = [str(f) for f in img_paths if f.name in np.loadtxt(TEST_LIST, dtype=str)]

    print("Total samples   :", len(img_paths))
    print("Testing samples :", len(test_fns))

    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=None,
    )

    network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    network = oem.utils.load_checkpoint(
        network,
        model_name="model.pth",
        model_dir="outputs",
    )

    network.eval().to(DEVICE)
    for i, idx in range(len(test_fns)):
        img, msk, fn = test_data[idx]

        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

        img = np.moveaxis(img.numpy(), 0, -1)
        msk = oem.utils.make_rgb(np.argmax(msk.numpy(), axis=0))
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))
