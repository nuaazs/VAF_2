# utils

import cfg
import importlib
ENCODE_MODEL_LIST = cfg.ENCODE_MODEL_LIST
from utils.log import logger
from utils.test import test_wav
emb_dict = {}
for model in ENCODE_MODEL_LIST:
    module = importlib.import_module(f"utils.encoder.{model}")
    emb_dict[model] = module.emb
# from utils.encoder.ECAPATDNN import emb as ECAPATDNN_emb
# from utils.encoder.CAMPP import emb as CAMPP_emb
# 打印 emb_dict 占用内存大小
import sys

def only_encode(wav_torch_raw, outinfo,action_type="test"):
    test_result = {}
    """Audio quality detection and encoding.
    Args:
        wav_torch_raw (torch 1D): wav data
        action_type (sting): Action type (register or test)

    Returns:
        Dict: Quality inspection results and audio characteristics
    """
    test_result = {}
    wav_torch_raw = wav_torch_raw.reshape(-1)
    if action_type == "register":
        min_length = cfg.MIN_LENGTH_REGISTER
    elif action_type == "test":
        min_length = cfg.MIN_LENGTH_TEST
    else:
        min_length = cfg.MIN_LENGTH_TEST  # sys.maxsize
    sr = cfg.ENCODE_SR
    max_score = 0
    mean_score = 0
    min_score = 1
    logger.info(f"\t\t wav_torch_raw shape: {wav_torch_raw.shape}")
    raw_wav_length = len(wav_torch_raw) / sr
    if raw_wav_length <= min_length:
        result = {
            "pass": False,
            "msg": f"encode Insufficient duration, the current duration is {len(wav_torch_raw) / sr}s. %d <= %d" % (
                raw_wav_length, min_length),
            "max_score": 0,
            "mean_score": 0,
            "min_score": 0,
            "err_type": 6,
            "before_score": None,
        }
        return result,outinfo
    # try:
    embeddings_dict = {}
    wav_torch_raw = wav_torch_raw.to("cuda:0")
    # assert wav_torch_raw.device == "cuda:0"
    wav_torch = wav_torch_raw.unsqueeze(0) # shape: [1, wav_length]
    for model in ENCODE_MODEL_LIST:
        outinfo.log_time(name=f"{model} before_encode_time")
        emb = emb_dict[model]
        embedding = emb.encode_batch(wav_torch)
        logger.info(f"\t\t {model} embedding shape: {embedding.shape}")
        embeddings_dict[model] = embedding.reshape(-1)
    result = {
        "pass": True,
        "msg": "Qualified.",
        "max_score": max_score,
        "before_score": None,
        "mean_score": mean_score,
        "min_score": min_score,
        "embeddings_dict": embeddings_dict,
        "err_type": 0,
    }
    return result,outinfo

def encode(wav_torch_raw, action_type,outinfo):
    test_result = {}
    """Audio quality detection and encoding.
    Args:
        wav_torch_raw (torch 1D): wav data
        action_type (sting): Action type (register or test)

    Returns:
        Dict: Quality inspection results and audio characteristics
    """
    test_result = {}
    wav_torch_raw = wav_torch_raw.reshape(-1)
    if action_type == "register":
        min_length = cfg.MIN_LENGTH_REGISTER
    elif action_type == "test":
        min_length = cfg.MIN_LENGTH_TEST
    else:
        min_length = cfg.MIN_LENGTH_TEST  # sys.maxsize
    sr = cfg.ENCODE_SR
    max_score = 0
    mean_score = 0
    min_score = 1
    logger.info(f"\t\t wav_torch_raw shape: {wav_torch_raw.shape}")
    raw_wav_length = len(wav_torch_raw) / sr
    if raw_wav_length <= min_length:
        result = {
            "pass": False,
            "msg": f"encode Insufficient duration, the current duration is {len(wav_torch_raw) / sr}s. %d <= %d" % (
                raw_wav_length, min_length),
            "max_score": 0,
            "mean_score": 0,
            "min_score": 0,
            "err_type": 6,
            "before_score": None,
        }
        return result,outinfo,test_result
    # try:
    embeddings_dict = {}
    wav_torch_raw = wav_torch_raw.to("cuda:0")
    wav_torch = wav_torch_raw.unsqueeze(0) # shape: [1, wav_length]
    for model in ENCODE_MODEL_LIST:
        outinfo.log_time(name=f"{model} before_encode_time")
        emb = emb_dict[model]
        embedding = emb.encode_batch(wav_torch)
        logger.info(f"\t\t {model} embedding shape: {embedding.shape}")
        embeddings_dict[model] = embedding.reshape(-1)
        outinfo.log_time(name=f"{model} real_encode_time")
        now_inbase,test_result[model] = test_wav(embedding.reshape(-1), black_limit=cfg.BLACK_TH[model],embedding_type=model)
        test_result[model]["inbase"] = now_inbase
        if not now_inbase:
            result = {
                "pass": False,
                "msg": f"Model {model} not hit, {test_result[model]}",
                "max_score": 0,
                "mean_score": 0,
                "min_score": 0,
                "err_type": 88, # 88 means not hit just exit
                "hit_scores": test_result[model],
                "before_score": None,
                "best_score": test_result[model]["best_score"],
                "blackbase_phone": test_result[model]["spk"],
                "top_10": test_result[model]["top_10"],
            }
            return result,outinfo,test_result
    
    result = {
        "pass": True,
        "msg": "Qualified.",
        "max_score": max_score,
        "before_score": None,
        "mean_score": mean_score,
        "min_score": min_score,
        "embeddings_dict": embeddings_dict,
        "err_type": 0,
    }
    return result,outinfo,test_result
