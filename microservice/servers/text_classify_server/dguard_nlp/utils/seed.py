import numpy as np
import torch
import random

def set_seed(seed_num):
    # set random seed
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    return None