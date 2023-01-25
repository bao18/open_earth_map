import os
import numpy as np
import torch

class_rgb_oem = {
    "unknown": [0, 0, 0],
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

class_grey_oem = {
    "unknown": 0,
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}


def make_rgb(a, grey_codes: dict = class_grey_oem, rgb_codes: dict = class_rgb_oem):
    """_summary_

    Args:
        a (numpy arrat): semantic label (H x W)
        rgd_codes (dict): dict of class-rgd code
        grey_codes (dict): dict of label code
    Returns:
        array: semantic label map rgb-color coded
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in grey_codes.items():
        out[a == v, 0] = rgb_codes[k][0]
        out[a == v, 1] = rgb_codes[k][1]
        out[a == v, 2] = rgb_codes[k][2]
    return out


def make_mask(a, grey_codes: dict = class_grey_oem, rgb_codes: dict = class_rgb_oem):
    """_summary_

    Args:
        a (numpy array): semantic map (H x W x 3)
        rgd_codes (dict): dict of class-rgd code
        grey_codes (dict): dict of label code

    Returns:
        array: semantic label map
    """
    out = np.zeros(shape=a.shape[:2], dtype="uint8")
    for k, v in rgb_codes.items():
        mask = np.all(np.equal(a, v), axis=-1)
        out[mask] = grey_codes[k]
    return out


def save_model(model, epoch: int, best_score: float, model_name: str, output_dir: str):
    """_summary_

    Args:
        model (_type_): torch model
        epoch (int): running epoch
        best_score (float): running best score
        model_name (str): model's name
        output_dir (str): path to the output directory
    """
    torch.save(
        {"state_dict": model.state_dict(), "epoch": epoch, "best_score": best_score},
        os.path.join(output_dir, model_name),
    )
    print("model saved")


def load_checkpoint(model, model_name: str, model_dir: str = "./"):
    """

    Args:
        checkpoint (path/str): Path to saved torch model
        model (object): torch model

    Returns:
        _type_: _description_
    """
    fn_model = os.path.join(model_dir, model_name)
    checkpoint = torch.load(fn_model, map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print(
        "Loaded model:{} (Epoch={}, Score={:.3f})".format(
            model_name, checkpoint["epoch"], checkpoint["best_score"]
        )
    )
    return model
