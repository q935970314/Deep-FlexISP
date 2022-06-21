import math
import os
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.functional import interpolate

from auxiliary.settings import DEVICE


def log_metrics(train_loss: float, val_loss: float, current_metrics: dict, best_metrics: dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    header = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=header, index=False)


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, Tensor]) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def normalize(img: np.ndarray) -> np.ndarray:
    max_int = 65535.0
    return np.clip(img, 0.0, max_int) * (1.0 / max_int)


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    """ Converts an image from height x width x channels to channels x height x width """
    return x.transpose(2, 0, 1)


def scale(x: Tensor) -> Tensor:
    """ Scales all values of a tensor between 0 and 1 """
    x = x - x.min()
    x = x / x.max()
    return x


def rescale(x: Tensor, size: Tuple) -> Tensor:
    """ Rescale tensor to image size for better visualization """
    return interpolate(x, size, mode='bilinear')


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()


def tvd(pred: Tensor, label: Tensor) -> Tensor:
    """
    Total Variation Distance (TVD) is a distance measure for probability distributions
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return (Tensor([0.5]) * torch.abs(pred - label)).sum()


def jsd(p: List, q: List) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2

