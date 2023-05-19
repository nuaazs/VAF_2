from flask import request

# utils
from utils.log import logger
from utils.preprocess import save_url
from utils.preprocess import save_file
from utils.preprocess import vad
from utils.preprocess import read_wav_data, resample
from utils.encoder import encode
from utils.register import register
from utils.test import test
from utils.info import OutInfo
from utils.preprocess import remove_fold_and_file
from utils.cmd import run_cmd
import speechbrain
import numpy as np
# cfg
import cfg


def pipeline(request_form, file_mode="file"):
    """get vad result

    Args:
        file_mode (string): "url" or "file"

    Returns:
        json: response
    """
    all_new_spkid = request_form["spkids"] #001,003,004 ..
    spkid_list = all_new_spkid.split(",")


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
        # new_file = request.files["wav_file"]
        # 获取所有的文件名
        filepath_list = []
        wav_list = []
        for key in spkid_list:
            logger.info(f"key:{key}")
            logger.info(f"filename:{request.files[key].filename}")
            # download all files
            new_file = request.files[key]
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
                filepath_list.append(filepath)
                wav_list.append(read_wav_data(filepath))
                logger.info(f"\t\t Download success. Filepath: {filepath}")
            except Exception as e:
                remove_fold_and_file(new_spkid)
                return outinfo.response_error(spkid=new_spkid, err_type=3, message=str(e))
    
    # 将wav_list中的所有wav文件的list合并成一个(n,L)的矩阵，n表示wav文件的个数，L表示最长的wav文件的长度，不足的部分用0补齐
    # step1:找到最长的wav文件的长度,并将所有的wav文件的长度补齐到最长的长度
    max_len = 0
    for wav in wav_list:
        if wav.shape[1] > max_len:
            max_len = wav.shape[1]
    for i in range(len(wav_list)):
        wav = wav_list[i]
        if wav.shape[1] < max_len:
            wav_list[i] = np.hstack((wav, np.zeros((wav.shape[0], max_len - wav.shape[1]))))
    # step2:将所有的wav文件的list合并成一个(n,L)的矩阵
    wav = np.vstack(wav_list)
    # step3:将wav文件的矩阵转换成torch.tensor
    wav = torch.from_numpy(wav)

    logger.info(f"\t\t Final wav shape: {wav.shape}")


    logger.info(f"\t\t Resample to 16k ... ")
    vad_result = resample(wav, cfg.SR, cfg.ENCODE_SR) # vad_result shape: (n,L)
    # =========================LOG TIME=========================
    outinfo.log_time(name="resample_16k")
    # STEP 3: Encoding
    logger.info(f"\t\t Start encoding ... ")
    encode_result = encode(wav_torch_raw=vad_result)
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
