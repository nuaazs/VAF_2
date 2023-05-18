# utils

import cfg
if cfg.USE_CAMPP:
    from utils.encoder.cam_val import emb
else:
    from utils.encoder import spkreg

# cfg



def encode(wav_torch_raw, action_type="test"):
    # TODO:降噪模块更新
    """Audio quality detection and encoding.

    Args:
        wav_torch_raw (torch 1D): wav data
        action_type (sting): Action type (register or test)

    Returns:
        Dict: Quality inspection results and audio characteristics
    """
    if action_type == "register":
        min_length = cfg.MIN_LENGTH_REGISTER
    elif action_type == "test":
        min_length = cfg.MIN_LENGTH_TEST
    else:
        min_length = cfg.MIN_LENGTH_TEST  # sys.maxsize
    # sr = cfg.SR
    sr = 16000
    max_score = 0
    mean_score = 0
    min_score = 1
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
        return result
    try:
        wav_torch = wav_torch_raw.unsqueeze(0)

        if cfg.USE_CAMPP:
            embedding = emb.forward(wav_torch).unsqueeze(0) #spkreg.encode_batch(wav_torch)
        else:
            embedding = spkreg.encode_batch(wav_torch)

        result = {
            "pass": True,
            "msg": "Qualified.",
            "max_score": max_score,
            "before_score": None,
            "mean_score": mean_score,
            "min_score": min_score,
            "tensor": embedding[0],  # encode_result[x_index][0],
            "err_type": 0,
        }
        return result
    except Exception as e:
        print(e)
        result = {
            "pass": False,
            "msg": f"encode error: {e}",
            "max_score": 0,
            "mean_score": 0,
            "min_score": 0,
            "err_type": 9,
            "before_score": None,
        }
        return result
