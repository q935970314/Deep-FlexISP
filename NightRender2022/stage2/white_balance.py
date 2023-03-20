import os
from time import time

import numpy as np
import torch.utils.data

from auxiliary.settings import DEVICE
import glob
from classes.core.Evaluator import Evaluator
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4

import math
import cv2 as cv
from copy import deepcopy
from tqdm import tqdm



def bayer_to_offsets(bayer_pattern="grbg"):
    """
    Transform bayer pattern to offsets in order 'RGrBGb'
    n.b. Support 'RGrBGb' bayer pattern only.
    Args:
        bayer_pattern: string, e.g. 'rggb'. Must be one of 'rggb', 'grbg', 'gbrg', 'bggr'

    Returns:
        offsets: packed raw image with 4 channels
    """
    bayer_pattern = bayer_pattern.lower()
    assert bayer_pattern in ['rggb', 'grbg', 'gbrg', 'bggr'], 'WRONG BAYER PATTERN!'
    if bayer_pattern == 'rggb':
        offsets = [[0,0],[0,1],[1,1],[1,0]]
    elif bayer_pattern == 'grbg':
        offsets = [[0,1],[0,0],[1,0],[1,1]]
    elif bayer_pattern == 'gbrg':
        offsets = [[1,0],[0,0],[0,1],[1,1]]
    else: #bayer_pattern == 'bggr':
        offsets = [[1,1],[0,1],[0,0],[1,0]]
    return offsets


def unpack_raw(rawim, bayer_pattern="grbg"):
    """
    Inverse of pack_raw_to_4ch.
    Args:
        rawim:
        bayer_pattern:

    Returns:

    """
    offsets = bayer_to_offsets(bayer_pattern)
    h, w, c = rawim.shape
    n = c // 4
    out = np.zeros_like(rawim).reshape((h * 2, w * 2, -1))
    out = np.squeeze(out)

    out[offsets[0][0]::2, offsets[0][1]::2] = np.squeeze(rawim[..., :n])
    out[offsets[1][0]::2, offsets[1][1]::2] = np.squeeze(rawim[..., n:2*n])
    out[offsets[2][0]::2, offsets[2][1]::2] = np.squeeze(rawim[..., 2*n:3*n])
    out[offsets[3][0]::2, offsets[3][1]::2] = np.squeeze(rawim[..., 3*n:])
    return out

def pack_raw_to_4ch(rawim, bayer_pattern="grbg"):
    """
    Pack raw to h/2 x w/2 x 4n with order "RGrBGb..."
    n.b. Support ordinary bayer pattern only.
    Args:
        rawim: numpy.ndarray in shape (h, w, ...)
        bayer_pattern: string, e.g. "rggb". Must be one of "rggb", "grbg", "gbrg", "bggr"

    Returns:
        out: packed raw image with 4n channels
    """
    offsets = bayer_to_offsets(bayer_pattern)
    if rawim.ndim == 2:
        rawim = np.expand_dims(rawim, axis=-1)
    rawim = np.concatenate((rawim[offsets[0][0]::2, offsets[0][1]::2],
                            rawim[offsets[1][0]::2, offsets[1][1]::2],
                            rawim[offsets[2][0]::2, offsets[2][1]::2],
                            rawim[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)
    return rawim


def main():
    MODEL_TYPE = "fc4_cwp"

    model = ModelFC4()

    data_root = "../stage1_output"
    input_list = glob.glob(os.path.join(data_root, "*.npy"))
    os.makedirs("../stage2_output", exist_ok=True)

    model_list = [0, 2]
    for inp_path in tqdm(input_list):
        rr, gg, bb = 0, 0, 0
        ori_img = np.load(inp_path).astype(np.float32)
    
        for model_indx in model_list:
            path_to_pretrained = os.path.join("../stage2/trained_models", MODEL_TYPE, "fold_{}".format(model_indx))
            model.load(path_to_pretrained)
            model.evaluation_mode()

            img = deepcopy(ori_img)
            img[:, :, 1] = (img[:, :, 1] + img[:, :, 3]) / 2
            img = img[:, :, :-1]

            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            img = torch.pow(img, 1.0 / 2.2)
            img = img.to(DEVICE)

            with torch.no_grad():
                pred, _, _ = model.predict(img, return_steps=True)
                pred = pred.detach().cpu().squeeze(0).numpy()

            # rgb gain
            r, g, b = pred

            r /= g
            b /= g
            g /= g

            r = 1./ r
            g = 1./ g
            b = 1./ b

            rr += r
            gg += g
            bb += b
          
        r = rr / len(model_list)
        g = gg / len(model_list)
        b = bb / len(model_list)

        ori_img[:, :, 0] *= r
        ori_img[:, :, 1] *= g
        ori_img[:, :, 2] *= b
        ori_img[:, :, 3] *= g

        ori_img = np.clip(ori_img, 0, 1)

        # CCM
        tmp_r = ori_img[:, :, 0]
        tmp_g = (ori_img[:, :, 1] + ori_img[:, :, 3]) / 2
        tmp_b = ori_img[:, :, 2]

        # fixed CCM matrix
        # 1.521689 -0.673763 0.152074 
        # -0.145724 1.266507 -0.120783 
        # -0.0397583 -0.561249 1.60100734
        out = np.zeros_like(ori_img)    
        out[:, :, 0] = 1.521689 * tmp_r + -0.673763 * tmp_g + 0.152074 * tmp_b
        out[:, :, 1] = -0.145724 * tmp_r + 1.266507 * ori_img[:, :, 1] + -0.120783 * tmp_b
        out[:, :, 2] = -0.0397583 * tmp_r + -0.561249 * tmp_g + 1.60100734 * tmp_b
        out[:, :, 3] = -0.145724 * tmp_r + 1.266507 * ori_img[:, :, 3] + -0.120783 * tmp_b

        np.save("../stage2_output/{}.npy".format(inp_path.split("/")[-1].split(".")[0]), out)

if __name__ == '__main__':
    main()
