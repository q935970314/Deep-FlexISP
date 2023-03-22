import os
import numpy as np

import cv2 
import glob
import json
from fractions import Fraction
from exifread.utils import Ratio

from tqdm import tqdm

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

def normalize(raw_image, black_level, white_level):
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
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def BPC(img):
    h, w = img.shape
    out = img.copy()
    for i in range(1, h-1):
        for j in range(1, w-1):
            patch = img[i-1:i+1, j-1:j+1].reshape(-1)

            Gh = np.partition(patch, -2)[-2]
            Gi = np.partition(patch, 1)[1]
            avg = (np.sum(patch) - (img[i, j] + Gh + Gi)) / 6
            dif = Gh - Gi
            
            if img[i, j] > (avg + dif) or img[i, j] < (avg - dif):
                out[i, j] = np.median(patch)

if __name__ == "__main__":
    inputs_path = "/data"
    img_list = sorted(glob.glob(inputs_path + "/*.png"))
    os.makedirs("../tmp", exist_ok=True)
    
    for path in tqdm(img_list):
        raw_image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        metadata = json_read(path.replace(".png", ".json"), object_hook=fraction_from_json)

        img = normalize(raw_image, metadata['black_level'], metadata['white_level']).astype(np.float32)
        img = img[:, 1:-1]

        np.save("../tmp/{}.npy".format(path.split("/")[-1].split(".")[0]), img)

