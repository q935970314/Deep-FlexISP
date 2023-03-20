import os
import numpy as np
import cv2 as cv
import cv2
from copy import deepcopy
import hdf5storage


import torch
import math
import glob
from tqdm import tqdm

import json
from fractions import Fraction
from scipy.io import loadmat
from PIL import Image, ImageOps
from exifread.utils import Ratio


def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data


def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object


def fractions2floats(fractions):
    floats = []
    for fraction in fractions:
        floats.append(float(fraction.numerator) / fraction.denominator)
    return floats


def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    if isinstance(color_matrix_1[0], Fraction):
        color_matrix_1 = fractions2floats(color_matrix_1)
    if isinstance(color_matrix_2[0], Fraction):
        color_matrix_2 = fractions2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    cam2xyz2 = np.linalg.inv(xyz2cam2)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * \
        demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    # xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
    #                      [-0.9692660, 1.8760108, 0.0415560],
    #                      [0.0556434, -0.2040259, 1.0572252]])
                         
    xyz2srgb = np.array([[1.521689, -0.673763, 0.152074],
                         [-0.145724, 1.266507, -0.120783],
                         [-0.0397583, -0.561249, 1.60100734]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis,
                          :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    # srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def apply_gamma(x):
    return x ** (1.0 / 3.2)
    x = x.copy()
    idx = x <= 0.0031308
    x[idx] *= 12.92
    x[idx == False] = (x[idx == False] ** (1.0 / 3.2)) * 1.055 - 0.055
    return x


def apply_tone_map(x, tone_mapping='Base'):
    if tone_mapping == 'Flash':
        return perform_flash(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Storm':
        return perform_storm(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Drago':
        tonemap = cv2.createTonemapDrago()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Mantiuk':
        tonemap = cv2.createTonemapMantiuk()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Reinhard':
        tonemap = cv2.createTonemapReinhard()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Linear':
        return np.clip(x/np.sort(x.flatten())[-50000], 0, 1)
    elif tone_mapping == 'Base':
        # return 3 * x ** 2 - 2 * x ** 3
        # tone_curve = loadmat('tone_curve.mat')
        x = np.clip(x, 0, 1)
        tone_curve = loadmat(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'tone_curve.mat'))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
        return tone_mapped_image
    else:
        raise ValueError(
            'Bad tone_mapping option value! Use the following options: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard"')


def illumination_parameters_estimation(current_image, illumination_estimation_option="sog"):
    ie_method = illumination_estimation_option.lower()
    if ie_method == "gw":
        ie = np.mean(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "sog":
        sog_p = 2.2
        ie = np.mean(current_image**sog_p, axis=(0, 1))**(1/sog_p)
        ie /= ie[1]
        return ie
    elif ie_method == "wp":
        ie = np.max(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "iwp":
        samples_count = 20
        sample_size = 20
        rows, cols = current_image.shape[:2]
        data = np.reshape(current_image, (rows*cols, 3))
        maxima = np.zeros((samples_count, 3))
        for i in range(samples_count):
            maxima[i, :] = np.max(data[np.random.randint(
                low=0, high=rows*cols, size=(sample_size)), :], axis=0)
        ie = np.mean(maxima, axis=0)
        ie /= ie[1]
        return ie
    else:
        raise ValueError(
            'Bad illumination_estimation_option value! Use the following options: "gw", "wp", "sog", "iwp"')


def white_balance(demosaic_img, as_shot_neutral):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)

    as_shot_neutral = np.asarray(as_shot_neutral)
    # transform vector into matrix
    if as_shot_neutral.shape == (3,):
        as_shot_neutral = np.diag(1./as_shot_neutral)

    assert as_shot_neutral.shape == (3, 3)

    white_balanced_image = np.dot(demosaic_img, as_shot_neutral.T)
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)

    return white_balanced_image



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


default_ccm = np.array([
[1,0,0],
[0,1,0],
[0,0,1],
])
def simple_isp(raw_4ch, bayer_pattern, rgb_gain=[1, 1, 1], CCM=default_ccm, gamma=2.2, as_uint8=True):
    # rgb gain
    raw_4ch[:, :, 0] = raw_4ch[:, :, 0] * rgb_gain[0]
    raw_4ch[:, :, 1] = raw_4ch[:, :, 1] * rgb_gain[1]
    raw_4ch[:, :, 2] = raw_4ch[:, :, 2] * rgb_gain[2]
    raw_4ch[:, :, 3] = raw_4ch[:, :, 3] * rgb_gain[1]
    
    raw_1ch = unpack_raw(raw_4ch, bayer_pattern)
    raw_1ch = raw_1ch.clip(0, 1)
    
    # cv::COLOR_BayerBG2BGR = 46, 对应RGGB拜耳排布
    # cv::COLOR_BayerGB2BGR = 47, 对应GRBG拜耳排布
    # cv::COLOR_BayerRG2BGR = 48, 对应BGGR拜耳排布
    # cv::COLOR_BayerGR2BGR = 49, 对应GBRG拜耳排布
    if bayer_pattern == "rggb":
        cvbayer2rgb = cv.COLOR_BayerBG2RGB_EA
    elif bayer_pattern == "grbg":
    	cvbayer2rgb = cv.COLOR_BayerGB2RGB_EA
    elif bayer_pattern == "bggr":
    	cvbayer2rgb = cv.COLOR_BayerRG2RGB_EA
    elif bayer_pattern == "gbrg":
    	cvbayer2rgb = cv.COLOR_BayerGR2RGB_EA
    
    rgb = cv.cvtColor((raw_1ch * 65535).round().astype(np.uint16), cvbayer2rgb)
    rgb = rgb.astype(np.float32) / 65535
    rgb = rgb.dot(CCM.T).clip(0, 1)
    # rgb = rgb * 4
    # rgb = rgb ** (1 / gamma)
    rgb = rgb.astype(np.float32)
    bgr = rgb[:, :, ::-1]
    if as_uint8:
        bgr = np.round(bgr * 255).clip(0, 255).astype(np.uint8)
    return bgr



data_root = "../tmp"
output_path = "../tmp"
input_list = glob.glob(os.path.join(data_root, "*.npy"))

for inp in tqdm(sorted(input_list)[::-1]):
    metadata = json_read("/data/{}.json".format(inp.split("/")[-1].split(".")[0]), object_hook=fraction_from_json)

    img = np.load(inp).astype(np.float32)
    img = np.clip(img, 0, 1)

    output = simple_isp(img, "bggr", as_uint8=False)
    # output = output[4:-4, 8:-8, :]
    output = output[:, :, ::-1]
    
    rgb_gain = illumination_parameters_estimation(output)
    output = white_balance(output, rgb_gain)

    output = apply_color_space_transform(output, metadata['color_matrix_1'], metadata['color_matrix_2'])
    output = transform_xyz_to_srgb(output) 

    output = apply_tone_map(output)

    output = apply_gamma(output).astype(np.float32)

    np.save("../tmp/{}.npy".format(inp.split("/")[-1].split(".")[0]), output[:, :, ::-1].astype(np.float32))
