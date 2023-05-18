# coding = utf-8
# @Time    : 2022-09-05  15:34:55
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: register.

import datetime

from utils.orm import to_database
from utils.orm import add_speaker
from utils.orm import to_database
from utils.orm import to_log
from utils.orm import get_embeddings
from utils.orm import to_database

import cfg


def register(outinfo, pool=False):
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
    add_success, phone_info = to_database(
        embedding=outinfo.embedding,
        spkid=outinfo.spkid,
        max_class_index=outinfo.class_num,
        log_phone_info=cfg.LOG_PHONE_INFO,
    )
    # print(f"Add success: {add_success}")
    if add_success:
        skp_info = {
            "name": "none",
            "phone": outinfo.spkid,
            "uuid": outinfo.oss_path,
            "hit": 0,
            "register_time": (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
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
            file_url=outinfo.oss_path,
            preprocessed_file_path=outinfo.preprocessed_file_path,
            valid_length=outinfo.after_length,
            show_phone=outinfo.show_phone,
            before_length=outinfo.before_length,
            after_length=outinfo.after_length
        )
        response = {
            "code": 2000,
            "status": "success",
            "err_type": 0,
            "err_msg": "Register success.",
            "name": "none",
            "phone": outinfo.spkid,
            "uuid": outinfo.oss_path,
            "hit": 0,
            "register_time": (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
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
        }
        return response
