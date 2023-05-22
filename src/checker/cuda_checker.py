import torch
from utils.log.log_wraper import logger

def check():
    # check if cuda is available
    logger.info("** -> Checking CUDA ... ")
    if torch.cuda.is_available():
        logger.info("** -> CUDA test: Pass ! ")
        return True,"CUDA is available"
    else:
        logger.info("** -> CUDA test: Error !!! ")
        return False,"ERROR!! CUDA is not available!"