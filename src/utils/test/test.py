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
from utils.orm import get_embeddings
from utils.test import test_wav
from utils.encoder import similarity
from utils.orm import to_database
from utils.preprocess import check_clip
from utils.orm import get_blackid
from utils.asr import get_asr_content
from utils.preprocess import remove_fold_and_file
from utils.encoder.cam_val import emb as double_model
import cfg
# log
from utils.log import logger

black_database = get_embeddings()

def save_audio(path: str,
               tensor: torch.Tensor,
               sampling_rate: int = 16000):
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
    is_inbase, check_result = test_wav(
        database=black_database,
        embedding=outinfo.embedding,
        black_limit=cfg.BLACK_TH,
    )

    hit_scores = check_result["best_score"]
    blackbase_phone = check_result["spk"]
    top_10 = check_result["top_10"]
    #=========================LOG TIME=========================
    outinfo.log_time("test_used_time")

    # save to redis test db
    add_success, phone_info = to_database(
        embedding=outinfo.embedding,
        spkid=outinfo.spkid,
        max_class_index=999,
        log_phone_info=cfg.LOG_PHONE_INFO,
        mode="test",
    )
    
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

        try:
            blackbase_id = get_blackid(blackbase_phone.split(",")[0])
        except Exception as e:
            logger.error(f"get blackbase_id error: {e}")
            blackbase_id = 0
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
            "blackbase_id": blackbase_id,
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
