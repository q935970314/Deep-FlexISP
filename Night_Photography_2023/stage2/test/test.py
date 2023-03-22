import os
from time import time

import numpy as np
import torch.utils.data

DEVICE_TYPE = "cuda:0"
from classes.core.Evaluator import Evaluator
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4


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

MODEL_TYPE = "fc4_cwp"
SAVE_PRED = False
SAVE_CONF = False
USE_TRAINING_SET = False


def main():
    import glob

    data_root = "/ssd/dataset/ntire22/night_render/nightimaging/raw_s3"
    input_list = glob.glob(os.path.join(data_root, "*.npy"))

    for inp_path in input_list:
        print(inp_path)
        img = np.load(inp_path)
        print(img.max(), img.min(), img.shape)
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze().to(DEVICE)

        for num_fold in range(3):
            path_to_pretrained = os.path.join("trained_models", MODEL_TYPE, "fold_{}".format(num_fold))
            model.load(path_to_pretrained)
            model.evaluation_mode()

            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(dataloader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    pred, _, conf = model.predict(img, return_steps=True)
                    print(pred)
                
                


if __name__ == '__main__':
    main()
