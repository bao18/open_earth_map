import os
import time
import warnings
import numpy as np
import torch
import oem
import torchvision
from pathlib import Path

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "OpenEarthMap_Mini"
    TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = 512
    N_CLASSES = 9
    LR = 0.0001
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(fns))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    train_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Rotate(),
            oem.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(IMG_SIZE),
        ],
    )

    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = oem.losses.JaccardLoss()

    max_score = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")

        train_logs = oem.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = oem.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )

        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="model2.pth",
                output_dir=OUTPUT_DIR,
            )

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
