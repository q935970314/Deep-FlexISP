import numpy as np
import cv2
import cv2 as cv

import os
import json
from fractions import Fraction
from exifread.utils import Ratio
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch
from copy import deepcopy


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

def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats

def normalize(raw_image, black_level, white_level, bl_fix=0):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        if type(black_level[0]) is Fraction:
            black_level = fractions2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

    if bl_fix:
        black_level_mask = black_level_mask + bl_fix

    normalized_image = (raw_image.astype(np.float32) - black_level_mask) / (white_level - black_level_mask)
    normalized_image = np.clip(normalized_image, 0, 1)
    return normalized_image

def bayer_to_offsets(bayer_pattern):
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
    
def pack_raw_to_4ch(rawim, offsets):
    """
    Pack raw to h/2 x w/2 x 4n with order "RGrBGb..." RGBG RGBG RGBG
    n.b. Support ordinary bayer pattern only.
    Args:
        rawim: numpy.ndarray in shape (h, w, ...)
        bayer_pattern: string, e.g. "rggb". Must be one of "rggb", "grbg", "gbrg", "bggr"

    Returns:
        out: packed raw image with 4n channels
    """


    if rawim.ndim == 2:
        rawim = np.expand_dims(rawim, axis=-1)
        rawim_pack = np.concatenate((rawim[offsets[0][0]::2, offsets[0][1]::2],
                                rawim[offsets[1][0]::2, offsets[1][1]::2],
                                rawim[offsets[2][0]::2, offsets[2][1]::2],
                                rawim[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)
    elif rawim.ndim ==3:
        frame_num = rawim.shape[2]
        rawim_pack = np.zeros((int(rawim.shape[0]/2), int(rawim.shape[1]/2), rawim.shape[2] * 4))
        for i in range(frame_num):
            rawim_temp = rawim[:,:,i]
            rawim_temp = np.expand_dims(rawim_temp, axis=-1)
            rawim_temp_pack = np.concatenate((rawim_temp[offsets[0][0]::2, offsets[0][1]::2],
                                              rawim_temp[offsets[1][0]::2, offsets[1][1]::2],
                                              rawim_temp[offsets[2][0]::2, offsets[2][1]::2],
                                              rawim_temp[offsets[3][0]::2, offsets[3][1]::2]), axis=-1)

            rawim_pack[:, :, i * 4:(i + 1) * 4] = rawim_temp_pack

    return rawim_pack



def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    if isinstance(color_matrix_1[0], Fraction):
        color_matrix_1 = fractions2floats(color_matrix_1)
    if isinstance(color_matrix_2[0], Fraction):
        color_matrix_2 = fractions2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * \
        demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image




def transform_xyz_to_srgb(xyz_image):
    xyz2srgb = np.array([[3.0799, -1.5372, -0.5428],
                         [-0.9212, 1.8760, 0.0452],
                         [0.0529, -0.2040, 1.1512]])   
    srgb_image = xyz2srgb[np.newaxis, np.newaxis,
                          :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def apply_gamma(x):
    # return x ** (1.0 / 2.4)
    x = x.copy()
    gray = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2GRAY)
    idx = gray <= 0.0031308
    idx = np.stack([idx, idx, idx], axis=-1)
    x[idx] *= 12.92
    x[idx == False] = (x[idx == False] ** (1.0 / 2.4)) * 1.055 - 0.055
    return x


def fix_orientation(image, orientation):
    type1 = "Horizontal (normal)"
    type2 = "Mirror horizontal"
    type3 = "Rotate 180"
    type4 = "Mirror vertical"
    type5 = "Mirror horizontal and rotate 270 CW"
    type6 = "Rotate 90 CW"
    type7 = "Mirror horizontal and rotate 90 CW"
    type8 = "Rotate 270 CW"

    # if type(orientation) is list:
    #     orientation = orientation[0]

    if orientation == type1:
        pass
    elif orientation == type2:
        image = cv.flip(image, 0)
    elif orientation == type3:
        image = cv.rotate(image, cv.ROTATE_180)
    elif orientation == type4:
        image = cv.flip(image, 1)
    elif orientation == type5:
        image = cv.flip(image, 0)
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == type6:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif orientation == type7:
        image = cv.flip(image, 0)
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif orientation == type8:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise

    return image


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
        tone_curve = loadmat(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'tone_curve.mat'))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
        return tone_mapped_image
    else:
        raise ValueError(
            'Bad tone_mapping option value! Use the following options: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard"')

def autocontrast_using_pil(img, cutoff=2):
    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    output_image = np.array(img_pil).astype(np.float32) / 255
    return output_image

def usm_sharp(img, weight=0.666, radius=10, threshold=10):
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


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


def get_net(NET, checkpoint_path, device):
    net = NET()
    load_net = torch.load(checkpoint_path, map_location="cpu")

    try:
        load_net = load_net['params']
    except:
        pass

    net.load_state_dict(load_net, strict=True)
    net = net.to(device)
    net = net.eval()
    return net



def pyrblend(img1, img2, mask):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mask = mask.astype(np.float32)
    
    # 图像向下取样, 构造高斯金字塔: [原图，下取样1次，下取样2次，下取样3次，下取样4次]
    levels = 5  # 高斯金字塔层数
    gaussPyr1, gaussPyr2 = [img1], [img2]  # 原始图像为高斯金字塔第 0 层
    for i in range(1, levels):  # 高斯金字塔共 5 层: 0,1,2,3,4
        gaussPyr1.append(cv2.pyrDown(gaussPyr1[i-1]))  # 计算第 i 层高斯金字塔
        gaussPyr2.append(cv2.pyrDown(gaussPyr2[i-1]))

    # 图像向上取样, 构造拉普拉斯金字塔 [第1层残差，第2层残差，第3层残差，第4层残差]
    lapPyr1, lapPyr2 = [], []  # 从最顶层开始恢复
    for i in range(levels-1):  # 拉普拉斯金字塔有 4 层: 0,1,2,3
        lapPyr1.append(gaussPyr1[i] - cv2.pyrUp(gaussPyr1[i+1]))
        lapPyr2.append(gaussPyr2[i] - cv2.pyrUp(gaussPyr2[i+1]))

    # 拉普拉斯金字塔左右拼接
    lapStack = []
    for i in range(levels-1):  # 拉普拉斯金字塔共 4 层: 0,1,2,3
        rows, cols, channel = lapPyr1[i].shape
        lmask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
        lmask = lmask[:, :, np.newaxis]
        
        splicing = lapPyr1[i] * (1-lmask) + lapPyr2[i]*lmask
        lapStack.append(splicing)

    # 由拼接后的Laplace金字塔恢复原图像
    rows, cols, channel = gaussPyr1[-1].shape  # 高斯金字塔顶层 G4:(32,32)
    lmask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
    lmask = lmask[:, :, np.newaxis]
    stackG4 = gaussPyr1[-1] * (1-lmask) + gaussPyr2[-1]*lmask # 拼接高斯金字塔顶层
    stackG3 = lapStack[3] + cv2.pyrUp(stackG4)  # stackG3:(64,64)
    stackG2 = lapStack[2] + cv2.pyrUp(stackG3)  # stackG2:(128,128)
    stackG1 = lapStack[1] + cv2.pyrUp(stackG2)  # stackG1:(256,256)
    stackG0 = lapStack[0] + cv2.pyrUp(stackG1)  # stackG0:(512,512)

    return stackG0