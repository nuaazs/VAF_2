from flask import request

# utils
from utils.log import logger
from utils.preprocess import save_url
from utils.preprocess import save_file
from utils.preprocess import vad
from utils.preprocess import read_wav_data, resample
from utils.encoder import encode,only_encode
from utils.register import register
from utils.test import test
from utils.info import OutInfo
from utils.preprocess.remove_fold import remove_fold_and_file
from utils.cmd import run_cmd
import speechbrain
import numpy as np
import torch
# cfg
import cfg


def pipeline(request_form, file_mode="file"):
    """get vad result

    Args:
        file_mode (string): "url" or "file"

    Returns:
        json: response
    """
    new_spkid = request_form["spkid"]

    call_begintime = request_form.get("call_begintime", "1999-02-18 10:10:10")
    call_endtime = request_form.get("call_endtime", "1999-02-18 10:10:10")
    show_phone = request_form.get("show_phone", new_spkid)
    channel = int(request_form.get("wav_channel", cfg.WAV_CHANNEL))
    do_vad = int(request_form.get("vad", 0))
    action_num = 3

    logger.info(f"# Donging VAD ... ")
    logger.info(f"# Action 3")
    logger.info(f"# ID: {new_spkid} ShowPhone: {show_phone}. ")

    new_spkid = request_form["spkid"]

    outinfo = OutInfo(action_num)
    outinfo.spkid = new_spkid
    outinfo.call_begintime = call_begintime
    outinfo.call_endtime = call_endtime
    outinfo.show_phone = show_phone

    # STEP 1: Get wav file.
    if file_mode == "file":
        logger.info(f"\t\t Downloading ...")
        new_file = request.files["wav_file"]
        if (new_file.filename.split('.')[-1] not in ["blob", "wav", "weba", "webm", "mp3", "flac", "m4a", "ogg", "opus",
                                                     "spx", "amr", "mp4", "aac", "wma", "m4r", "3gp", "3g2", "caf",
                                                     "aiff", "aif", "aifc", "au", "sd2", "bwf", "rf64"]):
            message = f"File type error. Only support wav, weba, webm, mp3, flac, m4a, ogg, opus, spx, amr, \
                mp4, aac, wma, m4r, 3gp, 3g2, caf, aiff, aif, aifc, au, sd2, bwf, rf64."
            return outinfo.response_error(spkid=new_spkid, err_type=2, message=message)
        try:
            if "blob" in new_file.filename:
                new_file.filename = "test.webm"
            filepath, outinfo.oss_path = save_file(file=new_file, spk=new_spkid, channel=channel)
            logger.info(f"\t\t Download success. Filepath: {filepath}")
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=3, message=str(e))
    elif file_mode == "url":
        new_url = request_form.get("wav_url")
        logger.info(f"\t\t Downloading from URL:{new_url} ...")
        try:
            filepath, outinfo.oss_path = save_url(url=new_url, spk=new_spkid, channel=channel)
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=4, message=str(e))

    outinfo.wav = read_wav_data(filepath)
    outinfo.wav = outinfo.wav.to("cuda:0")
    # print(outinfo.wav.shape)
    if len(outinfo.wav.shape) < 2:
        outinfo.wav = outinfo.wav.unsqueeze(0)
    outinfo.wav_vad = outinfo.wav
    if not do_vad:
        vad_result = {"wav_torch":outinfo.wav_vad}
    else:
        # STEP 2: VAD``
        # TO GPU
        
        # STEP 2: VAD
        logger.info(f"\t\t Doing VAD ... ")
        assert outinfo.wav.device == torch.device("cuda:0")
        vad_result = vad(wav=outinfo.wav, spkid=new_spkid, action_type="test", save=False,outinfo=outinfo)
        outinfo.after_length = vad_result["after_length"]
        outinfo.before_length = vad_result["before_length"]
        outinfo.wav_vad = vad_result["wav_torch"].clone()
        outinfo.wav_vad.to("cuda:0")
        assert outinfo.wav_vad.device == torch.device("cuda:0")
        outinfo.preprocessed_file_path = vad_result["preprocessed_file_path"]
        logger.info(f"\t\t VAD Success! Before: {vad_result['before_length']}, After: {vad_result['after_length']}")
        # =========================LOG TIME=========================
        outinfo.log_time(name="vad_used_time")

    logger.info(f"\t\t Resample to 16k ... ")
    vad_result["wav_torch"] = resample(vad_result["wav_torch"], cfg.SR, cfg.ENCODE_SR)
    # =========================LOG TIME=========================
    outinfo.log_time(name="resample_16k")
    # STEP 3: Encoding
    logger.info(f"\t\t Start encoding ... ")
    encode_result,outinfo = only_encode(wav_torch_raw=vad_result["wav_torch"],action_type="test",outinfo=outinfo)
    logger.info(f"\t\t End encoding ... ")
    # =========================LOG TIME=========================
    outinfo.log_time(name="encode_time")
    if encode_result["pass"]:
        embeddings_dict = encode_result["embeddings_dict"]
    else:
        remove_fold_and_file(new_spkid)
        return outinfo.response_error(spkid=new_spkid, err_type=encode_result["err_type"],
                                      message=encode_result["msg"])
    outinfo.embeddings_dict = embeddings_dict
    remove_fold_and_file(new_spkid)
    response = {}
    for _model_name in embeddings_dict.keys():
        response[_model_name] = embeddings_dict[_model_name].tolist()
    return response
