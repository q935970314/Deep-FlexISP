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


"""
* FC4 using confidence-weighted pooling (fc_cwp):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%
0	    1.73		1.47		1.50		0.50		3.53		4.20
1	    2.11		1.54		1.66		0.43		4.87		5.89
2	    1.92		1.45		1.52		0.52		4.22		5.66
Avg	    1.92		1.49		1.56		0.48		4.21		5.25
StdDev	0.19		0.05		0.09		0.05		0.67		0.92

* FC4 using summation pooling (fc_sum):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%	
0	    1.68        1.20	    1.35    	0.40	    3.71	    4.25
1	    2.11	    1.62	    1.68	    0.51	    4.74	    5.78
2	    1.79	    1.24	    1.35	    0.38	    4.21	    5.60
Avg	    1.86	    1.35	    1.46	    0.43	    4.22	    5.21
StdDev  0.22	    0.23	    0.19	    0.07	    0.52	    0.84
"""


def main():
    MODEL_TYPE = "fc4_cwp"

    model = ModelFC4()

    data_root = "../tmp"
    input_list = glob.glob(os.path.join(data_root, "*.npy"))

    model_list = [0, 1, 2]
    for inp_path in tqdm(sorted(input_list)):
        rr, gg, bb = 0, 0, 0
        ori_img = np.load(inp_path).astype(np.float32)
    
        for model_indx in model_list:
            path_to_pretrained = os.path.join("./trained_models", MODEL_TYPE, "fold_{}".format(model_indx))
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
        '''
        tmp_r = ori_img[:, :, 0]
        tmp_g = (ori_img[:, :, 1] + ori_img[:, :, 3]) / 2
        tmp_b = ori_img[:, :, 2]

        out = np.zeros_like(ori_img)
        out[:, :, 0] = 1.521689 * tmp_r + -0.673763 * tmp_g + 0.152074 * tmp_b
        out[:, :, 1] = -0.145724 * tmp_r + 1.266507 * ori_img[:, :, 1] + -0.120783 * tmp_b
        out[:, :, 2] = -0.0397583 * tmp_r + -0.561249 * tmp_g + 1.60100734 * tmp_b
        out[:, :, 3] = -0.145724 * tmp_r + 1.266507 * ori_img[:, :, 3] + -0.120783 * tmp_b
        '''

        np.save("../tmp/{}.npy".format(inp_path.split("/")[-1].split(".")[0]), ori_img)

if __name__ == '__main__':
    main()
