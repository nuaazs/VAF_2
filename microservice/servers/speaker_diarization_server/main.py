
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/08/16 15:59:25
@Author  :   Carry
@Version :   1.0
@Desc    :   对下载的当月音频进行话者分离及聚类操作，筛选出有效音频
'''


import datetime
import glob
import shutil
import numpy as np
import torchaudio
from tqdm import tqdm
import cfg
import os
import torch
from utils.oss.upload import upload_file
from tqdm.contrib.concurrent import process_map
from loguru import logger
from cluster_st import cluster_pipleline
from tools import send_request, extract_audio_segment, find_items_with_highest_value

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)


def pipeline(tmp_folder, filepath, spkid):
    # step1 VAD
    data = {"spkid": spkid, "length": 90}
    files = [('file', (filepath, open(filepath, 'rb')))]
    response = send_request(cfg.VAD_URL, files=files, data=data)
    vad_times = str(response['timelist'])

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
    if not mandarin_wavs:
        return None
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
        logger.info(f"After cluster {min_score}  < {cfg.CLUSTER_MIN_SCORE_THRESHOLD}. spkid:{spkid}.response:{response['scores']}")
        return None
    total_duration = 0
    for i in items.keys():
        total_duration += file_emb['length'][i]
    if total_duration < cfg.MIN_LENGTH_REGISTER:
        logger.info(f"After cluster total_duration:{total_duration} < {cfg.MIN_LENGTH_REGISTER}s. spkid:{spkid}.response:{response}")
        return None

    # Resample 16k
    selected_files = sorted(items.keys(), key=lambda x: x.split("/")[-1].replace(".wav", "").split("_")[0])
    resampled_waveform_li = []
    for file in selected_files:
        waveform, sample_rate = torchaudio.load(file.replace("local://", ""))
        if sample_rate == 8000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            resampled_waveform = resampler(waveform)
            resampled_waveform_li.append(resampled_waveform)
    audio_data = np.concatenate(resampled_waveform_li, axis=-1)
    file_selected_path = os.path.join(tmp_folder, f"{spkid}_selected.wav")
    torchaudio.save(file_selected_path, torch.from_numpy(audio_data), sample_rate=16000)

    selected_times = [d[_data.replace("local://", "")] for _data in selected_files]

    # step6 ASR
    text = ""
    # data = {"spkid": spkid, "postprocess": "1"}
    # files = [('wav_file', (filepath, open(filepath, 'rb')))]
    # response = send_request(cfg.ASR_URL, files=files, data=data)
    # if response.get('transcription') and response.get('transcription').get('text'):
    #     text = response['transcription']["text"]
    # else:
    #     logger.error(f"ASR failed. spkid:{spkid}.message:{response['message']}")

    # step7 NLP
    # nlp_result = classify_text(text)
    # logger.info(f"\t * -> 文本分类结果: {nlp_result}")

    # step8 上传OSS
    raw_url = upload_file(cfg.BUCKET_NAME, filepath, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(cfg.BUCKET_NAME, file_selected_path, f"{spkid}/{spkid}_selected.wav")

    return {
        "spkid": spkid,
        "raw_file_path": raw_url,
        "selected_url": selected_url,
        "asr_text": text,
        "selected_times": selected_times,
        # "nlp_result": nlp_result,
        "total_duration": total_duration,
        "record_month": record_month,
        "vad_times": vad_times
    }


def perprocess(filepath):
    """
    对单个音频进行处理
    通过筛选后的音频后续进行聚类
    """
    try:
        spkid = os.path.basename(filepath).split(".")[0].split('-')[-1]
        tmp_folder = f"/tmp/speaker_diarization/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)
        pipeline_result = pipeline(tmp_folder, filepath, spkid)
        if pipeline_result:
            with open(f"{need_cluster_records_path}", "a+") as f:
                f.write(str(pipeline_result)+'\n')
            logger.info(f"need_cluster_records:{pipeline_result}")
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. msg:{e}.")
    finally:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)


def get_last_id():
    if not os.path.exists('output/last_id.txt'):
        with open('output/last_id.txt', 'w') as f:
            f.write('0')
    with open('output/last_id.txt', 'r') as f:
        last_id = f.read()
    return last_id


def update_last_id(last_id):
    with open('output/last_id.txt', 'w') as f:
        f.write(str(last_id))


def main():
    wav_files = glob.glob("/datasets/changzhou/*.wav")
    # wav_files = glob.glob("./test_dataset/*.wav")
    logger.info(f"Total wav files: {len(wav_files)}")
    wav_files = sorted(wav_files)
    for i in tqdm(wav_files):
        record_num = os.path.basename(i).split("-")[-1].split('.')[0]  # 本地音频文件名
        # record_num = os.path.basename(i).split(".")[0]
        if int(record_num) < int(last_id):
            continue
        perprocess(i)

    with open(f"{need_cluster_records_path}", "r+") as f:
        need_cluster_records = f.readlines()

    need_cluster_records_li = []
    for i in need_cluster_records:
        i = eval(i)
        need_cluster_records_li.append(i)

    cluster_pipleline(need_cluster_records_li)

    new_last_id = os.path.basename(wav_files[-1]).split(".")[0]
    update_last_id(new_last_id)
    logger.info(f"New last id is: {new_last_id}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # 存储pipeline后需要聚类的音频信息文件
    need_cluster_records_path = "output/need_cluster_records.txt"
    # if os.path.exists(need_cluster_records_path):
    #     os.remove(need_cluster_records_path)

    need_cluster_records = []
    last_id = get_last_id()
    logger.info(f"Last id: {last_id}")
    currt_month = datetime.datetime.now().month
    record_month = str(currt_month)
    record_month = "8"
    main()
