
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/08/16 15:59:25
@Author  :   Carry
@Version :   1.0
@Desc    :   对下载的当月音频进行话者分离及聚类操作，筛选出有效音频
'''


import glob
import shutil
import numpy as np
import pymysql
import torchaudio
from tqdm import tqdm
import cfg
import os
import torch
from utils.oss.upload import upload_file
from tqdm.contrib.concurrent import process_map
from loguru import logger
from cluster_st import pipleline
from tools import send_request, extract_audio_segment, find_items_with_highest_value

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)


def pipeline(tmp_folder, filepath, spkid):
    # step1 VAD
    data = {"spkid": spkid, "length": 90}
    files = [('file', (filepath, open(filepath, 'rb')))]
    response = send_request(cfg.VAD_URL, files=files, data=data)

    # step2 截取音频片段
    output_file_li = []
    d = {}
    for idx, i in enumerate(response['timelist']):
        output_file = f"{tmp_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
        extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
        output_file_li.append(output_file)
        d[output_file] = (i[0]/1000, i[1]/1000)

    # step3 普通话过滤
    wav_files = ["local://"+i for i in output_file_li]
    data = {"spkid": spkid, "filelist": ",".join(wav_files)}
    response = send_request(cfg.LANG_URL, data=data)
    if response['code'] == 200:
        pass_list = response['pass_list']
        url_list = response['file_url_list']
        mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
    else:
        logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
        return None

    # step4 提取特征
    data = {"spkid": spkid, "filelist": ",".join(mandarin_wavs)}
    response = send_request(cfg.ENCODE_URL, data=data)
    if response['code'] == 200:
        file_emb = response['file_emb']
    else:
        logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
        return None

    # step5 聚类
    file_emb = file_emb[cfg.USE_MODEL_TYPE]
    data = {
        "emb_dict": file_emb["embedding"],
        "cluster_line": 3,
        "mer_cos_th": 0.7,
        "cluster_type": "spectral",  # spectral or umap_hdbscan
        "min_cluster_size": 1,
    }
    response = send_request(cfg.CLUSTER_URL, json=data)
    items, keys_with_max_value = find_items_with_highest_value(response['labels'])
    max_score = response['scores'][keys_with_max_value]['max']
    min_score = response['scores'][keys_with_max_value]['min']

    if min_score < cfg.CLUSTER_MIN_SCORE_THRESHOLD:
        logger.info(f"After cluster min_score  < {cfg.CLUSTER_MIN_SCORE_THRESHOLD}. spkid:{spkid}.response:{response['scores']}")
        return None
    total_duration = 0
    for i in items.keys():
        total_duration += file_emb['length'][i]
    if total_duration < cfg.MIN_LENGTH_REGISTER:
        logger.info(f"After cluster total_duration:{total_duration} < {cfg.MIN_LENGTH_REGISTER}s. spkid:{spkid}.response:{response}")
        return None
    selected_files = sorted(items.keys(), key=lambda x: x.split("/")[-1].replace(".wav", "").split("_")[0])
    audio_data = np.concatenate([torchaudio.load(file.replace("local://", ""))[0] for file in selected_files], axis=-1)
    file_selected_path = os.path.join(tmp_folder, f"{spkid}_selected.wav")
    torchaudio.save(file_selected_path, torch.from_numpy(audio_data), sample_rate=8000)

    selected_times = [d[_data.replace("local://", "")] for _data in selected_files]

    # step6 ASR
    text = ""
    data = {"spkid": spkid, "postprocess": "1"}
    files = [('wav_file', (filepath, open(filepath, 'rb')))]
    response = send_request(cfg.ASR_URL, files=files, data=data)
    if response.get('transcription') and response.get('transcription').get('text'):
        text = response['transcription']["text"]
    else:
        logger.error(f"ASR failed. spkid:{spkid}.message:{response['message']}")

    # step7 NLP
    # nlp_result = classify_text(text)
    # logger.info(f"\t * -> 文本分类结果: {nlp_result}")

    # step7 话术过滤
    # a, b = check_text(text)
    # if a == "正常":
    #     # todo 查找新话术逻辑
    #     return None

    # step8 上传OSS
    raw_url = upload_file(cfg.BUCKET_NAME, filepath, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(cfg.BUCKET_NAME, file_selected_path, f"{spkid}/{spkid}_selected.wav")

    return {
        "spkid": spkid,
        "raw_file_path": raw_url,
        "selected_url": selected_url,
        "asr_result": text,
        "selected_times": selected_times,
        # "nlp_result": nlp_result,
        "total_duration": total_duration,
    }


def insert_to_db(data):
    conn = pymysql.connect(
        host=cfg.MYSQL.get("host"),
        port=cfg.MYSQL.get("port"),
        db=cfg.MYSQL.get("db"),
        user=cfg.MYSQL.get("username"),
        passwd=cfg.MYSQL.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO check_for_speaker_diraization (`record_id`, `file_url`, `selected_url`, `asr_text`, `wav_duration`,`create_time`,`selected_times`, `record_month`) VALUES (%s, %s, %s, %s, %s,now(), %s, %s);"
        cursor.execute(sql, (data['spkid'], data['raw_file_path'], data['selected_url'], data['asr_result'],
                       data['total_duration'], str(data['selected_times']), record_month))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. spkid:{data['spkid']}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def main(i):
    try:
        spkid = os.path.basename(i).split(".")[0].split('-')[-1]
        tmp_folder = f"/tmp/speaker_diarization/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)
        pipeline_result = pipeline(tmp_folder, i, spkid)
        if pipeline_result:
            insert_to_db(pipeline_result)
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. msg:{e}.")
    finally:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    record_month = "7"
    # wav_files = glob.glob("/datasets/changzhou/*.wav")
    wav_files = glob.glob("./*.wav")
    logger.info(f"Total wav files: {len(wav_files)}")
    wav_files = sorted(wav_files)
    for i in tqdm(wav_files):
        main(i)

    pipleline(record_month)
