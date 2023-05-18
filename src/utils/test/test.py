# coding = utf-8
# @Time    : 2022-09-05  15:36:03
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: test.

import datetime
import time
import torchaudio
import torch

from utils.orm.query import add_hit_count
from utils.oss.upload import upload_file
from utils.orm import to_database
from utils.orm import to_log
from utils.orm import add_hit
from utils.test import test_wav
from utils.encoder import similarity
from utils.orm import to_database
from utils.preprocess import check_clip
from utils.orm import get_blackid
from utils.asr import get_asr_content
from utils.preprocess import remove_fold_and_file
import cfg

# log
from utils.log import logger

def save_audio(path: str,
               tensor: torch.Tensor,
               sampling_rate: int = 16000):
    tensor = tensor.reshape(-1)
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)

def test(outinfo, pool=False):
    """Audio reasoning, compare the audio features with the black library in full, and return the result.

    Args:
        embedding (tensor): audio features
        wav (tensor): audio data
        new_spkid (string): speaker ID
        max_class_index (int): pre-classify result
        oss_path (string): The file url on OSS
        self_test_result (dict): self-test results
        call_begintime (string): call start time
        call_endtime (string): call end time
        preprocessed_file_path (string): Preprocessed file address
        show_phone (string): Displayed phone number
        before_vad_length (float): Audio duration after VAD
        after_vad_length (float): Audio duration before VAD
        used_time (dict): Time spent on each module

    Returns:
        dict: inference result.
    """
    inbase_list = []
    check_result_dict = {}
    for model_name in outinfo.embeddings_dict.keys():
        is_inbase, check_result = test_wav(
            embedding=outinfo.embeddings_dict[model_name],
            black_limit=cfg.BLACK_TH[model_name],
        )
        inbase_list.append(is_inbase)
        check_result_dict[model_name]=check_result

        # save to redis test db
        add_success, phone_info = to_database(
            embedding=outinfo.embeddings_dict[model_name],
            spkid=outinfo.spkid,
            max_class_index=model_name,
            log_phone_info=cfg.LOG_PHONE_INFO,
            mode="test",
        )
        
    # if all inbase_list is True, then inbase
    is_inbase = all(inbase_list)
    hit_scores=""
    blackbase_phone=""
    for _model in check_result_dict.key():
        hit_scores += f"{_model}:{check_result_dict[_model]['best_score']},"
    for _model in check_result_dict.key():
        blackbase_phone += f"{_model}:{check_result_dict[_model]['best_id']},"
    top_10=""
    for _model in check_result_dict.key():
        top_10 += f"{_model}:{check_result_dict[_model]['top_10']},"

    #=========================LOG TIME=========================
    outinfo.log_time("test_used_time")

    
    
    if is_inbase:
        now_time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        timestr = now_time_str.replace(" ", "_").replace(":", "_")
        filename = outinfo.spkid + "_" + timestr + "_vad.wav"
        temp_save_path = f"{cfg.TEMP_PATH}/{outinfo.spkid}/{filename}"
        save_audio(temp_save_path, outinfo.wav_vad, sampling_rate=cfg.SR)
        outinfo.preprocessed_file_path = upload_file(
            bucket_name="preprocessed",
            filepath=temp_save_path,
            filename=filename,
            save_days=cfg.MINIO["test_save_days"],
        )

        # save to minio
        filename_raw = outinfo.spkid + "_" + timestr + ".wav"
        temp_save_path_raw = f"{cfg.TEMP_PATH}/{outinfo.spkid}/{filename_raw}"
        save_audio(temp_save_path_raw, outinfo.wav_vad, sampling_rate=cfg.SR)
        outinfo.raw_minio_file_url = upload_file(
            bucket_name="preprocessed",
            filepath=temp_save_path_raw,
            filename=filename,
            save_days=cfg.MINIO["test_save_days"],
        )

        # get asr content
        asr_content, hit_keyword, keyword = get_asr_content(outinfo.raw_minio_file_url, outinfo.spkid)
        hit_info = {
            "name": "none",
            "show_phone": outinfo.show_phone,
            "phone": outinfo.spkid,
            "file_url": outinfo.raw_minio_file_url,
            "hit_time": datetime.datetime.now(),
            "province": phone_info.get("province", ""),
            "city": phone_info.get("city", ""),
            "phone_type": phone_info.get("phone_type", ""),
            "area_code": phone_info.get("area_code", ""),
            "zip_code": phone_info.get("zip_code", ""),
            "call_begintime": outinfo.call_begintime,
            "call_endtime": outinfo.call_endtime,
            "class_number": 999,
            "preprocessed_file_path": outinfo.preprocessed_file_path,
            "blackbase_phone": blackbase_phone,
            "blackbase_id": 0,
            "hit_status": 1,
            "hit_scores": hit_scores,
            "top_10": top_10,
            "content_text": asr_content,
            "hit_keyword": hit_keyword,
            "keyword": keyword,
        }
        response = {
            "code": 2000,
            "status": "success",
            "inbase": is_inbase,
            "hit_scores": hit_scores,
            "blackbase_phone": blackbase_phone,
            "top_10": top_10,
            "err_msg": "null",
            "before_vad_length": outinfo.before_length,
            "after_vad_length": outinfo.after_length,
            "self_test_before_score": 1,
            "used_time": outinfo.used_time,
        }
        to_log(
            phone=outinfo.spkid,
            action_type=1,
            err_type=0,
            message=f"{is_inbase},{blackbase_phone},{hit_scores}",
            file_url=outinfo.oss_path,
            preprocessed_file_path=outinfo.preprocessed_file_path,
            valid_length=outinfo.after_length,
            show_phone=outinfo.show_phone,
            before_length=outinfo.before_length,
            after_length=outinfo.after_length
        )
        add_hit(hit_info, is_grey=False, after_vad_length=outinfo.after_length)
        add_hit_count(blackbase_phone)
        remove_fold_and_file(outinfo.spkid)
        #=========================LOG TIME=========================
        outinfo.log_time("to_database_used_time")
        return response
    else:
        response = {
            "code": 2000,
            "status": "success",
            "inbase": is_inbase,
            "err_msg": "null",
            "before_vad_length": outinfo.before_length,
            "after_vad_length": outinfo.after_length,
            "hit_scores": hit_scores,
            "blackbase_phone": blackbase_phone,
            "top_10": top_10,
            "used_time": outinfo.used_time,
        }
        to_log(
            phone=outinfo.spkid,
            action_type=1,
            err_type=99,
            message=f"False, Not in base, {blackbase_phone},{hit_scores}",
            file_url=outinfo.oss_path,
            preprocessed_file_path=outinfo.preprocessed_file_path,
            valid_length=outinfo.after_length,
            show_phone=outinfo.show_phone,
            before_length=outinfo.before_length,
            after_length=outinfo.after_length
        )
        remove_fold_and_file(outinfo.spkid)
        return response
