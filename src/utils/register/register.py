# coding = utf-8
# @Time    : 2022-09-05  15:34:55
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: register.

import datetime
from utils.html.wehtml import get_html

from utils.orm import to_database
from utils.orm import add_speaker
from utils.orm import to_database
from utils.orm import to_log
from utils.orm import get_embeddings
from utils.orm import to_database
from utils.oss.upload import upload_file

import cfg
from utils.preprocess.remove_fold import remove_fold_and_file
from utils.test.test import save_audio
from utils.log import logger


def upload_minio(outinfo, bucket_name):
    now_time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    timestr = now_time_str.replace(" ", "_").replace(":", "_")
    filename = outinfo.spkid + "_" + timestr + "_vad.wav"
    temp_save_path = f"{cfg.TEMP_PATH}/{outinfo.spkid}/{filename}"
    save_audio(
        temp_save_path,
        outinfo.wav_vad.clone().detach().cpu(),
        sampling_rate=cfg.ENCODE_SR,
    )
    outinfo.preprocessed_file_path = upload_file(
        bucket_name=bucket_name,
        filepath=temp_save_path,
        filename=filename,
        save_days=cfg.MINIO["register_save_days"],
    )

    # save to minio
    filename_raw = outinfo.spkid + "_" + timestr + "_raw.wav"
    temp_save_path_raw = f"{cfg.TEMP_PATH}/{outinfo.spkid}/{filename_raw}"
    save_audio(
        temp_save_path_raw, outinfo.wav.clone().detach().cpu(), sampling_rate=cfg.SR
    )
    outinfo.raw_minio_file_url = upload_file(
        bucket_name=bucket_name,
        filepath=temp_save_path_raw,
        filename=filename_raw,
        save_days=cfg.MINIO["register_save_days"],
    )


def register(outinfo, need_check=0, pool=False):
    """Audio registration, write voiceprint library.

    Args:
        embedding (tensor): audio features
        wav (tensor): audio data
        new_spkid (string): speaker ID
        max_class_index (int): Pre-classification results
        oss_path (string): The path of the file on oss
        self_test_result (dict): self-test results
        call_begintime (string): call start time
        call_endtime (string): call end time
        preprocessed_file_path (string): Preprocessed file save path
        show_phone (string): Displayed phone number
        before_vad_length (float): Audio duration before VAD
        after_vad_length (float): Audio duration after VAD
        used_time (dict): Time spent on each module

    Returns:
        dict: Registration result
    """

    """
    # ASR
    try:
        html_content, vue_k, vue_kwd = get_html(outinfo.raw_minio_file_url, outinfo.spkid)
    except Exception as e:
        html_content = ""
        vue_k = ""
        vue_kwd = ""
        logger.error(f"get_html error: {e}")

    # check asr
    if not vue_k:
        return outinfo.response_error(
            spkid=outinfo.spkid,
            err_type=11,
            message=f"Request html error.html_content:{html_content}",
        )
    """
    if need_check:
        # save to minio
        upload_minio(outinfo, bucket_name="check")

        response = {
            "code": 200,
            "status": "success",
            "message": "Step 1 check pass,please go to step 2 by manually.",
            "before_vad_length": outinfo.before_length,
            "after_vad_length": outinfo.after_length,
            "file_url": outinfo.raw_minio_file_url,
            "preprocessed_file_url": outinfo.preprocessed_file_path,
        }
        remove_fold_and_file(outinfo.spkid)
        return response
    else:
        for model_name in outinfo.embeddings_dict.keys():
            # save to redis test db
            add_success, phone_info = to_database(
                embedding=outinfo.embeddings_dict[model_name],
                spkid=outinfo.spkid,
                max_class_index=model_name,
                log_phone_info=cfg.LOG_PHONE_INFO,
            )

        if add_success:
            upload_minio(outinfo, bucket_name="black")
            skp_info = {
                "name": "none",
                "phone": outinfo.spkid,
                "uuid": outinfo.raw_minio_file_url,
                "hit": 0,
                "register_time": (datetime.datetime.now()).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "province": phone_info.get("province", ""),
                "city": phone_info.get("city", ""),
                "phone_type": phone_info.get("phone_type", ""),
                "area_code": phone_info.get("area_code", ""),
                "zip_code": phone_info.get("zip_code", ""),
                "call_begintime": outinfo.call_begintime,
                "call_endtime": outinfo.call_endtime,
                "max_class_index": outinfo.class_num,
                "preprocessed_file_path": outinfo.preprocessed_file_path,
                "show_phone": outinfo.show_phone,
            }
            add_speaker(skp_info, after_vad_length=outinfo.after_length)
            to_log(
                phone=outinfo.spkid,
                action_type=2,
                err_type=0,
                message=f"Register success.",
                file_url=outinfo.raw_minio_file_url,
                preprocessed_file_path=outinfo.preprocessed_file_path,
                valid_length=outinfo.after_length,
                show_phone=outinfo.show_phone,
                before_length=outinfo.before_length,
                after_length=outinfo.after_length,
            )
            response = {
                "code": 2000,
                "status": "success",
                "err_type": 0,
                "err_msg": "Register success.",
                "name": "none",
                "phone": outinfo.spkid,
                "uuid": outinfo.raw_minio_file_url,
                "hit": 0,
                "register_time": (datetime.datetime.now()).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "province": phone_info.get("province", ""),
                "city": phone_info.get("city", ""),
                "phone_type": phone_info.get("phone_type", ""),
                "area_code": phone_info.get("area_code", ""),
                "zip_code": phone_info.get("zip_code", ""),
                "call_begintime": outinfo.call_begintime,
                "call_endtime": outinfo.call_endtime,
                "max_class_index": outinfo.class_num,
                "preprocessed_file_path": outinfo.preprocessed_file_path,
                "show_phone": outinfo.show_phone,
                "before_vad_length": outinfo.before_length,
                "after_vad_length": outinfo.after_length,
                "used_time": outinfo.used_time,
                "gender": outinfo.gender_result,
            }
            remove_fold_and_file(outinfo.spkid)
            return response
