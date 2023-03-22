import re

import numpy as np
import torch


# --- Determinism (for reproducibility) ---

def make_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cuda:0"


def get_device() -> torch.device:
    if DEVICE_TYPE == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(DEVICE_TYPE))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(DEVICE_TYPE))
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))


DEVICE = get_device()

# --- Model ---

# If set to False, a simpler summation pooling will be used
USE_CONFIDENCE_WEIGHTED_POOLING = True
if not USE_CONFIDENCE_WEIGHTED_POOLING:
    print("\n WARN: confidence-weighted pooling option is set to False \n")

# Input size
TRAIN_IMG_W, TRAIN_IMG_H = 512, 512
TEST_IMG_W, TEST_IMG_H = 0, 0
