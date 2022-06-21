import os
import tensorflow as tf
import numpy as np

from unet import sc_net_1f

from copy import deepcopy
from tqdm import tqdm


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


if __name__ == "__main__":
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, None, None, 4])
    out_image = sc_net_1f(in_image)

    #load checkpoint
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    latest_checkpoint = tf.train.latest_checkpoint("../stage1/checkpoint/sc_net_1f_pretrained_model")

    if latest_checkpoint:
        print('loaded ' + latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
    else:
        raise SystemExit('No checkpoint!')

    offsets = bayer_to_offsets("bggr")

    # read data and run
    data_root = "../stage0_output"
    os.makedirs("../stage1_output", exist_ok=True)
    input_list = os.listdir(data_root)
    
    for inp_path in tqdm(input_list):
        input_full = np.load(os.path.join(data_root, inp_path)).astype(np.float32)
        # input_full = np.pad(input_full, ((0, 6), (0, 4)), 'reflect')
        input_full = pack_raw_to_4ch(input_full, offsets)

        input_full = np.clip(input_full, 0.0, 1.0)
        input_full = np.expand_dims(input_full, axis=0)
        input_full = np.float32(input_full)
        ori_inp = deepcopy(input_full)
        
        clip_min = max(np.mean(input_full)*3, 0.9)
        # clip_min = max(np.mean(input_full)*2, 0.2)
        input_full = np.clip(input_full, 0, clip_min)

        output = sess.run(out_image, feed_dict={in_image: input_full})
       
        output = ori_inp + output
        output = np.clip(output, 0, 1)
        output = np.squeeze(output)

        np.save("../stage1_output/{}.npy".format(inp_path.split(".")[0]), output)

