import os
import csv
import numpy as np
from pathlib import Path

np.random.seed(100)

if __name__ == "__main__":
    OEMDIR = "/home/bruno/riken_openearthmap/LULC-RIKEN/integrated"
    R = 0.2

    img_paths = [f for f in Path(OEMDIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [
        str(f) for f in img_paths if f.name in np.loadtxt("data/train.txt", dtype=str)
    ]
    val_fns = [
        str(f) for f in img_paths if f.name in np.loadtxt("data/val.txt", dtype=str)
    ]
    test_fns = [
        str(f) for f in img_paths if f.name in np.loadtxt("data/test.txt", dtype=str)
    ]

    for k, fns in {"train": train_fns, "val": val_fns, "test": test_fns}.items():
        f = open(f"data/{k}_mini.txt", "w")
        writer = csv.writer(f)

        imgs = np.asarray([x.split("/")[-1] for x in fns])
        cities = np.asarray([x.split("/")[-3] for x in fns])

        cnts, vals = np.unique(cities, return_counts=True)

        for c, n in zip(cnts, vals):

            idx_city = np.nonzero(cities == c)[0]
            idxs = idx_city[np.argsort(np.random.rand(n))[: int(R * n)]]

            if len(idxs) > 0:
                for idx in idxs:
                    writer.writerow([imgs[idx]])

        f.close()
