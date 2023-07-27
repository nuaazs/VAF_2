# coding = utf-8
# @Time    : 2022-09-05  09:47:51
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Register and identify interfaces.
import os

import numpy as np
import speechbrain
import torchaudio
from flask import request
import torch

# utils
from utils.orm import check_spkid
from utils.log import logger
from utils.preprocess import save_url
from utils.preprocess import save_file
from utils.preprocess import vad
from utils.preprocess import read_wav_data, resample
from utils.encoder import encode, only_encode
from utils.register import register
from utils.test import test
from utils.info import OutInfo
from utils.preprocess.remove_fold import remove_fold_and_file
from utils.cmd import run_cmd
from utils.orm import to_log
import cfg
from utils.test.scores import get_similarity

if cfg.FILTER_MANDARIN:
    from utils.preprocess.mandarin_filter import filter_mandarin


def general(request_form, file_mode="url", action_type="test"):
    """_summary_

    Args:
        request_form (form):{'spkid': '1', 'wav_url': 'http://xxxxx/1.wav', 'wav_channel': 1}
        get_type (str, optional): url or file. Defaults to "url".
        action (str, optional): register or test. Defaults to "test".

    Returns:
        _type_: response: {'code':2000,'status':"success",'err_type': '1', 'err_msg': ''}
        * err_type:
            # 1. spkid repeat
            # 2. The file format is incorrect
            # 3. File parsing error
            # 4. File download error
            # 5. The file has no valid data (very poor quality)
            # 6. The valid duration of the file does not meet the requirements
            # 7. The file quality detection does not meet the requirements
            #    (the environment is noisy or there are multiple speakers)
    """

    new_spkid = request_form["spkid"]
    call_begintime = request_form.get("call_begintime", "1999-02-18 10:10:10")
    call_endtime = request_form.get("call_endtime", "1999-02-18 10:10:10")
    show_phone = request_form.get("show_phone", new_spkid)
    channel = int(request_form.get("wav_channel", cfg.WAV_CHANNEL))
    only_vad = int(request_form.get("only_vad", cfg.ONLY_VAD))
    show_vad_list = int(request_form.get("show_vad_list", cfg.SHOW_VAD_LIST))
    use_fbank = int(request_form.get("use_fbank", cfg.USE_FBANK))
    gender = int(request_form.get("gender", cfg.GENDER_CLASSIFY))
    only_gender = int(request_form.get("only_gender", cfg.GENDER_CLASSIFY_ONLY))
    need_check = int(request_form.get("need_check", cfg.NEED_CHECK))
    logger.info(f"# ID: {new_spkid} ShowPhone: {show_phone}. ")

    if action_type == "register":
        action_num = 2
        register_date = request_form.get("register_date", "20200101")
    if action_type == "test":
        action_num = 1

    outinfo = OutInfo(action_num)
    outinfo.spkid = new_spkid
    outinfo.call_begintime = call_begintime
    outinfo.call_endtime = call_endtime
    outinfo.show_phone = show_phone

    if cfg.CHECK_DUPLICATE:
        if check_spkid(new_spkid) and action_type == "register":
            message = f"ID: {new_spkid} already exists. Deny registration."
            return outinfo.response_error(spkid=new_spkid, err_type=1, message=message)

    # STEP 1: Get wav file.
    if file_mode == "file":
        logger.info(f"\t\t Downloading ...")
        new_file = request.files["wav_file"]
        if new_file.filename.split(".")[-1] not in [
            "blob", "wav", "weba", "webm", "mp3", "flac", "m4a", "ogg", "opus", "spx", "amr", "mp4", "aac",
                "wma", "m4r", "3gp", "3g2", "caf", "aiff", "aif", "aifc", "au", "sd2", "bwf", "rf64",
        ]:
            message = f"File type error. Only support wav, weba, webm, mp3, flac, m4a, ogg, opus, spx, amr, \
                mp4, aac, wma, m4r, 3gp, 3g2, caf, aiff, aif, aifc, au, sd2, bwf, rf64."
            return outinfo.response_error(spkid=new_spkid, err_type=2, message=message)
        try:
            if "blob" in new_file.filename:
                new_file.filename = "test.webm"
            filepath, outinfo.oss_path = save_file(
                file=new_file, spk=new_spkid, channel=channel
            )
            logger.info(f"\t\t Download success. Filepath: {filepath}")
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=3, message=str(e))
    elif file_mode == "url":
        new_url = request_form.get("wav_url")
        logger.info(f"\t\t Downloading from URL:{new_url} ...")
        try:
            filepath, outinfo.oss_path = save_url(
                url=new_url, spk=new_spkid, channel=channel
            )
        except Exception as e:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=4, message=str(e))
    # =========================LOG TIME=========================
    outinfo.log_time(name="download_used_time")
    logger.info(f"\t\t Download success. Filepath: {filepath}")
    if cfg.ONLY_USE_ENERGY_VAD:
        # STEP 2: VAD
        logger.info(f"\t\t Doing VAD ... ")
        save = False
        if only_vad:
            save = True
        # filepath
        vad_result = vad_result = vad(
            bin_file_path=filepath,
            spkid=new_spkid,
            action_type=action_type,
            save=save,
            outinfo=outinfo,
        )

        outinfo.after_length = vad_result["after_length"]
        outinfo.before_length = vad_result["before_length"]
        outinfo.wav_vad = vad_result["wav_torch"].clone()
        outinfo.wav_vad = outinfo.wav_vad.to("cuda:0")

        assert outinfo.wav_vad.device == torch.device("cuda:0")

        outinfo.preprocessed_file_path = vad_result["preprocessed_file_path"]
        logger.info(
            f"\t\t EnergyBase VAD Success! Before: {vad_result['before_length']}, After: {vad_result['after_length']}"
        )
        # =========================LOG TIME=========================
        outinfo.log_time(name="vad_used_time")

    else:
        outinfo.wav = read_wav_data(wav_filepath=filepath)
        # TO GPU
        outinfo.wav = outinfo.wav.to("cuda:0")

        # STEP 2: VAD
        logger.info(f"\t\t Doing VAD ... ")
        save = False
        if only_vad:
            save = True

        assert outinfo.wav.device == torch.device("cuda:0")

        vad_result = vad(wav=outinfo.wav, spkid=new_spkid, action_type=action_type, save=save, outinfo=outinfo)
        outinfo.after_length = vad_result["after_length"]
        outinfo.before_length = vad_result["before_length"]
        outinfo.wav_vad = vad_result["wav_torch"].clone()
        outinfo.wav_vad.to("cuda:0")

        assert outinfo.wav_vad.device == torch.device("cuda:0")

        outinfo.preprocessed_file_path = vad_result["preprocessed_file_path"]
        logger.info(f"\t\t VAD Success! Before: {vad_result['before_length']}, After: {vad_result['after_length']}")
        # =========================LOG TIME=========================
        outinfo.log_time(name="vad_used_time")

    if only_vad:
        response = {
            "output_vad_file_path": outinfo.preprocessed_file_path,
            "vad_used_time": outinfo.used_time["vad_used_time"],
            "vad_before_length": outinfo.before_length,
            "vad_after_length": outinfo.after_length,
        }
        if show_vad_list:
            response["wav_torch"] = outinfo.wav_vad.tolist()
            response["mask"] = vad_result["boundaries"].tolist()
        remove_fold_and_file(new_spkid)
        return response

    if use_fbank:
        logger.info(f"\t\t Doing fbank ... ")
        fbank = speechbrain.lobes.features.Fbank(sample_rate=8000, n_mels=80)
        vad_result["wav_torch"] = vad_result["wav_torch"].unsqueeze(0)
        fbank_data = fbank(vad_result["wav_torch"])

        response = {
            "fbank_data": np.array(fbank_data).tolist(),
            "vad_used_time": outinfo.used_time["vad_used_time"],
            "vad_before_length": outinfo.before_length,
            "vad_after_length": outinfo.after_length,
        }
        remove_fold_and_file(new_spkid)
        return response
    assert outinfo.wav_vad.device == torch.device("cuda:0")

    after_wav_length = len(outinfo.wav_vad.reshape(-1)) / cfg.SR
    if after_wav_length < cfg.MIN_LENGTH_TEST:
        remove_fold_and_file(new_spkid)
        return outinfo.response_error(spkid=new_spkid, err_type=6, message=f"Length too short, length:{after_wav_length}")

    outinfo.wav_vad = resample(outinfo.wav_vad, cfg.SR, cfg.ENCODE_SR)

    if len(outinfo.wav_vad.shape) > 1:
        outinfo.wav_vad = outinfo.wav_vad[0]
    logger.info(f"\t\t vad_result wav_torch shape {outinfo.wav_vad.shape} ")
    # =========================LOG TIME=========================
    outinfo.log_time(name="resample_16k")

    if cfg.FILTER_MANDARIN:
        assert outinfo.wav_vad.device == torch.device("cuda:0")
        is_mandarin, lang, score = filter_mandarin(wavdata=outinfo.wav_vad, score_threshold=cfg.FILTER_MANDARIN_TH)  # True,result[3][0],score
        if not is_mandarin:
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=7, message=f"Language is {lang}, score is {score}")
        # =========================LOG TIME=========================
        outinfo.log_time(name="filter_mandarin")

    # STEP 3: Encoding
    assert outinfo.wav_vad.device == torch.device("cuda:0")
    if action_num == 1:
        encode_result, outinfo, test_result = encode(wav_torch_raw=outinfo.wav_vad, action_type=action_type, outinfo=outinfo)
    else:
        encode_result, outinfo = only_encode(wav_torch_raw=outinfo.wav_vad, action_type="register", outinfo=outinfo)
    # =========================LOG TIME=========================
    outinfo.log_time(name="encode_time")
    if encode_result["pass"]:
        embeddings_dict = encode_result["embeddings_dict"]
    elif action_num == 1 and encode_result["err_type"] == 88:
        response = {
            "code": 2000, "status": "success", "inbase": False, "err_msg": "null", "before_vad_length": outinfo.before_length,
            "after_vad_length": outinfo.after_length, "hit_scores": encode_result["best_score"],
            "blackbase_phone": encode_result["blackbase_phone"],
            "top_10": encode_result["top_10"],
            "used_time": outinfo.used_time, "gender": outinfo.gender_result
        }
        to_log(
            phone=outinfo.spkid, action_type=1, err_type=88, message=encode_result["msg"], file_url=outinfo.oss_path,
            preprocessed_file_path=outinfo.preprocessed_file_path, valid_length=outinfo.after_length,
            show_phone=outinfo.show_phone, before_length=outinfo.before_length, after_length=outinfo.after_length
        )
        remove_fold_and_file(new_spkid)
        return response
    else:
        remove_fold_and_file(new_spkid)
        return outinfo.response_error(spkid=new_spkid, err_type=encode_result["err_type"], message=encode_result["msg"])

    outinfo.embeddings_dict = embeddings_dict
    
    # STEP 4: Test or Register
    if action_num == 1:
        logger.info(f"\t\t Testing ... ")
        outinfo.class_num = 999
        return test(outinfo, test_result)
    elif action_num == 2:
        logger.info(f"\t\t Registering ... ")
        outinfo.class_num = register_date
        all_inbase = []
        msg_list = []
        for model in cfg.ENCODE_MODEL_LIST:
            now_inbase, msg = get_similarity(outinfo.embeddings_dict[model], black_limit=cfg.BLACK_TH[model], embedding_type=model)
            all_inbase.append(now_inbase)
            msg_list.append(msg)
        if all(all_inbase):
            remove_fold_and_file(new_spkid)
            return outinfo.response_error(spkid=new_spkid, err_type=4, message=f"ID: {new_spkid} already exists. msg:{msg_list}.")
        else:
            return register(outinfo, need_check)
