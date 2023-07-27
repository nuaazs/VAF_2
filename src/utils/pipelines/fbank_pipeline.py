import os
import torchaudio
from flask import request

# utils
from utils.orm import check_spkid
from utils.log import logger
from utils.preprocess import save_url
from utils.preprocess import save_file
from utils.preprocess import vad
from utils.preprocess import read_wav_data,resample
from utils.encoder import encode
from utils.register import register
from utils.test import test
from utils.info import OutInfo
from utils.preprocess.remove_fold import remove_fold_and_file
from utils.cmd import run_cmd
import speechbrain
import numpy as np
# cfg
import cfg

def pipeline(request_form,request_files,file_mode="file"):
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
            filepath, outinfo.oss_path = save_file(file=new_file, spk=new_spkid,channel=channel)
            logger.info(f"\t\t Download success. Filepath: {filepath}")
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=3, message=str(e))
    elif file_mode == "url":
        new_url = request_form.get("wav_url")
        logger.info(f"\t\t Downloading from URL:{new_url} ...")
        try:
            filepath, outinfo.oss_path = save_url(url=new_url, spk=new_spkid,channel=channel)
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=4, message=str(e))
    
    outinfo.wav = read_wav_data(filepath)
    # STEP 2: VAD
    logger.info(f"\t\t Doing VAD ... ")
    if len(outinfo.wav.shape) == 1:
        outinfo.wav = outinfo.wav.unsqueeze(0)
    vad_result = vad(wav=outinfo.wav, spkid=new_spkid,save=True)
    outinfo.after_length = vad_result["after_length"]
    outinfo.before_length = vad_result["before_length"]
    outinfo.wav_vad = vad_result["wav_torch"]
    outinfo.preprocessed_file_path = vad_result["preprocessed_file_path"]
    logger.info(f"{vad_result['wav_torch'].shape}")
    logger.info(f"\t\t VAD Success! Before: {vad_result['before_length']}, After: {vad_result['after_length']}")
    #=========================LOG TIME=========================
    outinfo.log_time(name="vad_used_time")
    # get fbank from 8k wav data vad_result["wav_torch"]
    # STEP 3: Get 80 fbank by speechbrain.lobes.features.Fbank
    logger.info(f"\t\t Doing fbank ... ")
    fbank= speechbrain.lobes.features.Fbank(sample_rate=8000,n_mels=80)
    # print(vad_result["wav_torch"].shape)
    vad_result["wav_torch"] = vad_result["wav_torch"].unsqueeze(0)
    fbank_data = fbank(vad_result["wav_torch"])

    response = {'fbank_data': np.array(fbank_data).tolist(),
                "vad_used_time":outinfo.used_time["vad_used_time"],
                "vad_before_length":outinfo.before_length,
                "vad_after_length":outinfo.after_length,
                }
    remove_fold_and_file(new_spkid)
    return response