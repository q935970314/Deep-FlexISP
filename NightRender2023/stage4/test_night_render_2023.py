import os
import numpy as np
import cv2 as cv
from copy import deepcopy
import hdf5storage


import torch
import math
import glob
from tqdm import tqdm

import json
from fractions import Fraction

def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data

def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object

def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv.flip(image, 0)
    elif orientation == 3:
        image = cv.rotate(image, cv.ROTATE_180)
    elif orientation == 4:
        image = cv.flip(image, 1)
    elif orientation == 5:
        image = cv.flip(image, 0)
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv.flip(image, 0)
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    return image

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


def get_net(NET, checkpoint_path, device):
    net = NET()
    load_net = torch.load(checkpoint_path, map_location="cpu")

    try:
        load_net = load_net['params']
    except:
        load_net = load_net['state_dict_model']

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)

    net.load_state_dict(load_net, strict=True)
    net = net.to(device)
    net = net.eval()
    return net


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


from network import MWRCAN as NET
checkpoint_path = "./100.pth"
model = get_net(NET, checkpoint_path, device)

data_root = "../tmp"
input_list = glob.glob(os.path.join(data_root, "*.npy"))

output_path = "../tmp"
os.makedirs(output_path, exist_ok=True)


for inp in tqdm(sorted(input_list)):
    img = np.load(inp).astype(np.float32)
    img = np.pad(img, ((0, 24), (0, 16), (0, 0)), 'reflect')

    img = torch.from_numpy(img.copy().transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    
    # output = np.pad(output, ((4, 4), (1, 1), (0, 0)), 'reflect')
    output = output[:-6, :-4, :]

    output = output * 255.
    output = output.round()  
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    metadata = json_read("/data/{}.json".format(inp.split("/")[-1].split(".")[0]), object_hook=fraction_from_json)
    output = fix_orientation(output, metadata["orientation"])
    
    # cv.imwrite(os.path.join(output_path, "{}.jpg".format(inp.split("/")[-1].split(".")[0])), output, [cv.IMWRITE_JPEG_QUALITY, 100])
    cv.imwrite(os.path.join(output_path, "{}.png".format(inp.split("/")[-1].split(".")[0])), output, [int(cv.IMWRITE_PNG_COMPRESSION), 0])

    
    
