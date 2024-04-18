import os

import multiprocessing as mp
import numpy as np
import cv2 as cv
from tqdm import tqdm

from glob import glob
from utils import *

from grayness_index import GraynessIndex

import torch
import torch.nn.functional as F

from time import time

from network_raw_denoise import sc_net_1f
from network import MWRCANv4 as NET
from classes.fc4.ModelFC4 import ModelFC4

def load_img(img_path):
    meta_all = {}
    meta_all['img_path'] = img_path

    # load meta
    metadata = json_read(img_path.replace(".png", ".json"), object_hook=fraction_from_json)
    meta_all['meta'] = metadata

    # load image
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    meta_all['img'] = img

    return meta_all




def pre_process(meta_all):
    img = meta_all['img']

    metadata = meta_all['meta']
    cfa_pattern = metadata['cfa_pattern']
    cfa_pattern_ = ""
    for tt in cfa_pattern:
        if tt == 0:
            cfa_pattern_ += "r"
        elif tt == 1:
            cfa_pattern_ += "g"
        elif tt == 2:
            cfa_pattern_ += "b"
        else:
            raise

    offsets = bayer_to_offsets(cfa_pattern_)
    img = pack_raw_to_4ch(img, offsets)
    
    if img.shape[0] != 768 and img.shape[1] != 1024:
        img = cv.resize(img, (1024, 768), interpolation=cv.INTER_AREA) # RGB
    
    bl_fix = np.clip((float(metadata["noise_profile"][0])-0.005) * 1000, 0, 10)
    img = normalize(img, metadata['black_level'], metadata['white_level'], bl_fix).astype(np.float32)

    noise_profile = float(metadata["noise_profile"][0])
    noise_list = [0.00025822882, 0.000580020745, 0.00141667975, 0.00278965863, 0.00347614807]

    if noise_profile < 0.005:
        if noise_profile < noise_list[0]:
            weight1 = noise_profile / noise_list[0]
            final_lsc = lsc_npy[0] * weight1
            linear_idx1, linear_idx2 = 0, 0
        elif noise_profile > noise_list[-1]:
            final_lsc = lsc_npy[-1]
            linear_idx1, linear_idx2 = -1, -1
        else:
            for idx, nn in enumerate(noise_list):
                if noise_profile < nn:
                    linear_idx1 = idx - 1
                    linear_idx2 = idx
                    break

            weight1 = (noise_profile - noise_list[linear_idx1]) / (noise_list[linear_idx2] - noise_list[linear_idx1])
            weight2 = 1-weight1
            final_lsc = lsc_npy[linear_idx1] * weight1 + lsc_npy[linear_idx2] * weight2

        ones = np.ones_like(final_lsc)
        final_lsc = final_lsc * 0.6 + ones * 0.4
        final_lsc[:, :512, :] = final_lsc[:, 1024:511:-1, :]
        
        img = img * final_lsc
    
    img = np.clip(img, 0.0, 1.0)
    meta_all["img"] = img
    
    rgb_gain = metadata['as_shot_neutral']
    ra, ga, ba = rgb_gain
    ra, ga, ba = 1/ra, 1/ga, 1/ba
    
    meta_all['r_gains'] = [ra]
    meta_all['g_gains'] = [ga]
    meta_all['b_gains'] = [ba]
    
    return meta_all



def raw_denoise(results):   
    checkpoint_path = "checkpoint/raw_denoise.pth"
    device = torch.device("cuda")
    model = get_net(sc_net_1f, checkpoint_path, device)

    for meta_all in tqdm(results):
        img = meta_all['img']
        
        img = np.expand_dims(img, axis=0)
        ori_inp = img.copy()
        
        clip_min = max(np.mean(img)*3, 0.9)
        img = np.clip(img, 0, clip_min)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).cuda()

        with torch.no_grad():
            output = model(img)
            
        output = output.detach().cpu().numpy().transpose(0, 2, 3, 1)

        img = ori_inp + output
        img = np.clip(img, 0, 1)
        img = np.squeeze(img)
        
        meta_all['img'] = img



def predict_white_balance(results):   
    model = ModelFC4()
    for model_index in [0, 1, 2]:
        path_to_pretrained = os.path.join("./trained_models", "fc4_cwp", "fold_{}".format(model_index))
        model.load(path_to_pretrained)
        model.evaluation_mode()

        for meta_all in tqdm(results):
            img = meta_all['img'].copy()
            img[:, :, 1] = (img[:, :, 1] + img[:, :, 3]) / 2
            img = img[:, :, :-1]

            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
            img_tmp = torch.pow(img, 1.0 / 2.2)

            with torch.no_grad():
                pred = model.predict(img_tmp, return_steps=False)
            pred = pred.detach().cpu().squeeze(0).numpy()

            # rgb gain
            r, g, b = pred

            r /= g
            b /= g
            g /= g

            r = 1./ r
            g = 1./ g
            b = 1./ b
            
            meta_all['r_gains'].append(r)
            meta_all['g_gains'].append(g)
            meta_all['b_gains'].append(b)


def convert_to_rgb(meta_all):
    img = meta_all['img']
    img[:, :, 1] = (img[:, :, 1] + img[:, :, 3]) / 2
    img = img[:, :, :-1]

    
    # WB
    r_gains = sorted(meta_all['r_gains'])
    b_gains = sorted(meta_all['b_gains'])

    r_final = (r_gains[0] + r_gains[1] + r_gains[2]) / 3
    g_final = 1
    b_final = (b_gains[1] + b_gains[2] + b_gains[3]) / 3

    img[:, :, 0] *= r_final
    img[:, :, 1] *= g_final
    img[:, :, 2] *= b_final

    img = np.clip(img, 0, 1)
    
    
    # CC
    img = apply_color_space_transform(img, color_matrix, color_matrix)


    # convert RGB
    img = transform_xyz_to_srgb(img)


    # shading fix
    if float(meta_all['meta']["noise_profile"][0]) > 0.005:   
        lsc_m = lsc ** ((float(meta_all['meta']["noise_profile"][0])-0.005) * 100)
        lsc_inv = 1 / lsc
        lsc_inv = np.mean(lsc_inv, axis=-1, keepdims=True)
        
        gray = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2GRAY)
        gray = gray[:, :, np.newaxis]

        lsc_inv = lsc_inv * np.clip(gray*10, 0, 1) * np.clip((2 - (float(meta_all['meta']["noise_profile"][0])-0.005) * 100), 1, 2)
        lsc_inv = np.clip(lsc_inv, 0.4, 1)

        img = img * lsc_inv + gray * (1-lsc_inv)
        img = img / lsc_m


    # tonemaping
    img = apply_tone_map(img)


    # gamma
    img = apply_gamma(img).astype(np.float32)
    img = np.clip(img, 0, 1)
    

    # contrast enhancement
    mm = np.mean(img)
    meta_all['mm'] = mm
    if mm <= 0.1:
        pass
    elif float(meta_all['meta']["noise_profile"][0]) > 0.01: 
        yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        y, u, v = cv.split(yuv)
        y = autocontrast_using_pil(y)
        yuv = np.stack([y, u, v], axis=-1)
        rgb = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
        rgb = np.clip(rgb, 0, 1)

        img = img * 0.5 + rgb * 0.5
        
        img = np.clip(img*255, 0, 255).round().astype(np.uint8)

        if float(meta_all['meta']["noise_profile"][0]) > 0.02:
            noise_params = 6
        else:
            noise_params = 3
        
        img = cv.fastNlMeansDenoisingColored(img, None, noise_params, noise_params, 7, 21)
        img = img.astype(np.float32) / 255.
        
        img = usm_sharp(img)
    else: 
        img = autocontrast_using_pil(img)


    # gamma again
    img = np.clip(img, 0, 1)
    img_con = img ** (1/1.5)
    gray = np.max(img_con, axis=-1, keepdims=True) # - 0.1
    gray = np.clip(gray, 0.3, 1)
    img = img_con * gray + img * (1-gray)


    # AWB again
    img = img[:, :, ::-1] # BGR
    gi = GraynessIndex()
    pred_illum = gi.apply(img)
    r, g, b = pred_illum
    pred_illum = pred_illum / g
    r, g, b = pred_illum
    if r < 1:
        img = white_balance(img, pred_illum) # BGR
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 1)  # RGB
    
    
    # fix orientation
    img = fix_orientation(img, meta_all['meta']["orientation"])
    
    meta_all['img'] = img # RGB
    return meta_all


def nn_enhancement(results):
    checkpoint_path1 = "checkpoint/nn_enhance.pth"
    device = torch.device("cuda")
    model = get_net(NET, checkpoint_path1, device)


    for meta_all in tqdm(results):
        # mm = meta_all['mm']
        # if mm <= 0.1 or float(meta_all['meta']["noise_profile"][0]) > 0.01:
        #     meta_all['img'] = meta_all['img'] * 255
        #     continue

        img = meta_all['img']
        img = img.astype(np.float32)
        img = torch.from_numpy(img.copy().transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            img = model(img)
            # img = img

        img = img.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)

        img = img * 255.
        img = img.round()  
        img = img.astype(np.uint8)
        
        meta_all['img'] = img # RGB U8


def post_process(meta_all):
    # color fix
    img = meta_all['img'] # RGB U8
    
    
    # increase saturation
    increment=0.5
    ori_img = img.copy() # RGB U8

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    _, L, S = cv2.split(hls)
    S = S / 255.
    
    img = img.astype(np.float32)

    temp = increment + S
    mask_2 = temp >  1  # 大于1的位置
    alpha_1 = S
    alpha_2 = 1 - increment
    alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
    L = L[:, :, np.newaxis]
    alpha = alpha[:, :, np.newaxis]
    
    alpha = 1/alpha -1 

    img = img + (img - L) * alpha
    
    img = np.clip(img, 0, 255)
    
    ori_img = ori_img.astype(np.float32)
    mask = ori_img[:, :, 2] / 255.
    # mask = np.max(ori_img, axis=-1) / 255.
    mask = mask[:, :, np.newaxis]
    mask = np.clip(mask - 0.1, 0, 1)
    img = img * mask + ori_img * (1-mask)
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    
    
    
    # decrease saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = hsv.astype(np.float32)
    
    # 绿 青
    mmax = 105
    mmin = 40
    
    alpha = 1   # 越大效果越猛, 误伤越多
    beta = 4 # 越大效果越猛
    gamma = 0.1    # 越小效果越猛

    mid = mmin + ((mmax - mmin) / 2)
    green_weight = np.abs(hsv[:, :, 0] - mid) / ((mmax - mmin)/2)
    green_weight = np.clip(green_weight, 0, 1)
    # green_weight = np.tanh(green_weight/alpha)
    green_weight = green_weight**beta + gamma
    green_weight = np.clip(green_weight, 0, 1)
    
    green_weight = cv2.blur(green_weight, (11, 11))
    hsv[:, :, 2] = hsv[:, :, 2] * green_weight 
    
    
    # 紫 洋红
    mmax = 180
    mmin = 130
    
    alpha = 1   # 越大效果越猛, 误伤越多
    beta = 8
        # 越大效果越猛
    gamma = -0.5    # 越小效果越猛

    mid = mmin + ((mmax - mmin) / 2)
    green_weight = np.abs(hsv[:, :, 0] - mid) / ((mmax - mmin)/2)
    green_weight = np.clip(green_weight, 0, 1)
    # green_weight = np.tanh(green_weight/alpha)
    green_weight = (green_weight**beta + gamma) * 2
    green_weight = np.clip(green_weight, 0, 1)
    
    green_weight = cv2.blur(green_weight, (11, 11))
    hsv[:, :, 2] = hsv[:, :, 2] * green_weight 
    
    
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.round().astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HLS2RGB) # RGB U8
    img = np.clip(img, 0, 255)


    
    
    
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    meta_all['img'] = img # RGB U8
    return meta_all
    


def sky_enhancement(results):      
    model_path = "./checkpoint/sky_seg.pt" 
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    for meta_all in tqdm(results):
        if float(meta_all['meta']["noise_profile"][0]) >= 0.005:
            continue
        
        ori_img = meta_all['img'].copy().astype(np.float32) # RGB 0-255 U8
        
        img = ori_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        # 天空分割
        scene_image = img.copy().astype(np.float32)  # 0-255, bgr
        
        
        img = img / 255.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 减少色温
        lab[:,:,1] = lab[:,:,1] - (lab[:,:,2] + 127) * 0.03
        lab[:,:,2] = lab[:,:,2] - (lab[:,:,2] + 127) * 0.1
        # 将图像从LAB空间转换回BGR空间
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = img * 255
        img = np.clip(img, 0, 255).round().astype(np.float32)

        
        
        img_mean = 0
        img_std = 255.0
        size = (512, 512)
        img_h , img_w = img.shape[:2]

        img = cv2.resize(img, size)
        img = (img - img_mean) / img_std
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).cuda()

        with torch.no_grad():
            mask = model(img)
        
        mask = mask.detach().cpu()
        mask = mask.permute((0,3,1,2))
        mask = F.interpolate(mask, 
                size=[img_h , img_w],
                mode='bilinear')
        mask = mask[0].permute((1,2,0))
        sky_mask = torch.argmax(mask, axis=2).numpy().astype(np.float32)
        
        if sky_mask.max() < 0.1:
            continue

        
        # 
        img = ori_img.copy()  # RGB
        mask = img[:, :, 2] - np.max(img[:, :, :2], axis=-1)
        mask[sky_mask==0]=0
        a = np.sum(mask)
        b = np.sum(sky_mask)
        ratio_blue = a/b
        # print(meta_all['img_path'], "blue ratio", ratio_blue)

        # 非蓝天
        if ratio_blue < 10:
            img = ori_img.copy()
            mask = np.mean(img[:, :, :2], axis=-1)
            mask[sky_mask==0]=0
            a = np.sum(mask)
            b = np.sum(sky_mask)
            ratio_light = a/b
            # print(meta_all['img_path'], "light ratio", ratio_light)
            
            # 暗天空，压暗
            if ratio_light<50: 
                img = ori_img.copy()
                img = img * 0.88
                img = np.clip(img, 0, 255) # RGB
            # 中等亮度，提亮
            elif ratio_light < 200:
                img = ori_img.copy()
                img = img * 1.1
                img = np.clip(img, 0, 255) # RGB
            else:
                pass

            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1]* 0.4
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # 蓝天
        else:
            # LAB
            img = ori_img.copy()
            img = img / 255.
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

            # 减少色温
            lab[:,:,1] = lab[:,:,1] - (lab[:,:,2] + 127) * 0.03
            lab[:,:,2] = lab[:,:,2] - (lab[:,:,2] + 127) * 0.1

            # 将图像从LAB空间转换回BGR空间
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            img = img * 255
            img = np.clip(img, 0, 255).round().astype(np.float32)
        

        sky_image = img.copy().astype(np.float32) # 0-255, RGB
        sky_image = cv2.cvtColor(sky_image, cv2.COLOR_RGB2BGR) # BGR 0-255 F32
        
        sky_mask_ori = sky_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        sky_mask_ori = cv2.erode(sky_mask_ori, kernel)

        sky_mask_ori = sky_mask_ori > 0.9
        
        if np.sum(sky_mask_ori) > 0:
            h, w = sky_mask.shape

            sky_mask = cv2.resize(sky_mask, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sky_mask = cv2.dilate(sky_mask, kernel)
            sky_mask_blur = cv2.blur(sky_mask, (21, 21))
            sky_mask_blur[sky_mask>0.5] = sky_mask[sky_mask>0.5]
            sky_mask = sky_mask_blur
            sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            sky_mask = np.clip(sky_mask, 0.1, 1)      

            sky_area_img = np.zeros_like(sky_image)
            sky_area_img[sky_mask_ori] = sky_image[sky_mask_ori]
            sky_area_img = cv2.cvtColor(sky_area_img, cv2.COLOR_BGR2GRAY)
            
            sky_area_img_mean = np.sum(sky_area_img) / np.sum(sky_mask_ori)
            if sky_area_img_mean > 20:       
                res = pyrblend(scene_image, sky_image, sky_mask)
                res = np.clip(res, 0, 255) # 0-255, bgr
                
                res = res.round().astype(np.uint8)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # RGB 0-255 U8
                meta_all['img'] = res



def post_process2(meta_all):
    # PIL
    img = meta_all['img'].copy() # RGB U8
    
    img = img.astype(np.float32) / 255.
    
    yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    y, u, v = cv.split(yuv)
    y = autocontrast_using_pil(y)
    yuv = np.stack([y, u, v], axis=-1)
    rgb = cv.cvtColor(yuv, cv.COLOR_YUV2RGB)
    rgb = np.clip(rgb, 0, 1)

    img = rgb
    
    img = np.clip(img*255, 0, 255)# .round().astype(np.uint8)  # RGB


    ori_img = meta_all['img'].copy().astype(np.float32)
    mask = np.mean(ori_img, axis=-1) / 255.
    mask = mask[:, :, np.newaxis]
    mask = np.clip(mask - 0.1, 0, 1)
    img = img * mask + ori_img * (1-mask)
    img = np.clip(img, 0, 255)

    img = img.round().astype(np.uint8)
    meta_all['img'] = img
    
    return meta_all


def save_jpg(meta_all):  
    img = meta_all['img'] # RGB U8
    out_path = os.path.join(output_path, meta_all['img_path'].split("/")[-1].split(".")[0] + ".jpg")

    cv.imwrite(out_path, img[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 100])
    


if __name__ == "__main__":   
    num_worker = 4
    
    all_time = time()
    
    input_path = "/data"
    output_path = "/data"
    # input_path = "/ssd/ntire24/nightrender/data/data"
    # output_path = "/ssd/ntire24/nightrender/data/data"
    os.makedirs(output_path, exist_ok=True)

    
    # load img
    s_time = time()   
    input_list = sorted(glob(os.path.join(input_path, "*.png")))# [:4]
    
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            results = list(tqdm(pool.imap(load_img, input_list), total=len(input_list)))
    else:
        results = []
        for p in tqdm(input_list):
            results.append(load_img(p))
    load_time = time()-s_time
    print("load_img time is: ", load_time)
    

    # preprocess
    s_time = time()
    iso_list = [50, 125, 320, 640, 800]
    lsc_npy = [np.load("./lsc_npy/{}.npy".format(iso)) for iso in iso_list]
    
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            results = list(tqdm(pool.imap(pre_process, results), total=len(results)))
    else:
        for r in tqdm(results):
            r = pre_process(r)
    del lsc_npy
    print("pre_process time is: ", time()-s_time)


    # raw denoise
    s_time = time()
    raw_denoise(results)
    print("raw_denoise time is: ", time()-s_time)
    

    # awb
    s_time = time()
    predict_white_balance(results)
    print("predict_white_balance time is: ", time()-s_time)


    # convert
    s_time = time()
    color_matrix = [1.06835938, -0.29882812, -0.14257812, -0.43164062,  1.35546875,  0.05078125, -0.1015625,   0.24414062,  0.5859375]
    lsc = np.load("lsc.npy")
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            results = list(tqdm(pool.imap(convert_to_rgb, results), total=len(results)))
    else:
        for r in tqdm(results):
            r = convert_to_rgb(r)
    del lsc
    print("convert_to_rgb time is: ", time()-s_time)


    # NN_enhancement
    s_time = time()
    nn_enhancement(results)
    print("nn_enhancement time is: ", time()-s_time)
    
    
    
    # colorfix & sat enhance
    s_time = time()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            results = list(tqdm(pool.imap(post_process, results), total=len(results)))
    else:
        for r in tqdm(results):
            r = post_process(r)
    print("post_process time is: ", time()-s_time)

    

    # sky_enhancement
    s_time = time()
    sky_enhancement(results)
    print("sky_enhancement time is: ", time()-s_time)
    
    
    
    # PIL autocontrast
    s_time = time()
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            results = list(tqdm(pool.imap(post_process2, results), total=len(results)))
    else:
        for r in tqdm(results):
            r = post_process2(r)
    print("post_process2 time is: ", time()-s_time)
    


    # save jpg
    s_time = time()
    if num_worker > 1:
        with mp.Pool(num_worker) as pool:
            _ = list(tqdm(pool.imap(save_jpg, results), total=len(results)))
    else:
        for r in tqdm(results):
            save_jpg(r)
    save_time = time()-s_time
    print("save_jpg time is: ", save_time)


    total_time = time()-all_time   
    total_time_without_load_save = total_time - load_time - save_time
    print("per image inference time (without load and save) is: ", total_time_without_load_save / len(results), "s")
