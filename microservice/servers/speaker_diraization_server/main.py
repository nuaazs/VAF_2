
from collections import Counter
import glob
import subprocess
import numpy as np
import pymysql
import torchaudio
from tqdm import tqdm
import cfg
from pydub import AudioSegment
import os
import torch
from utils.oss.upload import upload_file
import requests
from tqdm.contrib.concurrent import process_map


from loguru import logger

logger.add("log/file_{time}.log", rotation="500 MB", encoding="utf-8",
           enqueue=True, compression="zip", backtrace=True, diagnose=True)


vad_url = "http://192.168.3.169:5005/energy_vad/file"  # VAD
lang_url = "http://192.168.3.169:5002/lang_classify"  # 语种识别
encode_url = "http://192.168.3.169:5001/encode"  # 提取特征
cluster_url = "http://192.168.3.169:5011/cluster"  # cluster
asr_url = "http://192.168.3.169:5000/transcribe/file"  # ASR
use_model_type = "ECAPATDNN"


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(
            method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Request failed: spkid:{data['spkid']}. msg:{e}")
        return None


def extract_audio_segment(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def find_items_with_highest_value(dictionary):
    value_counts = Counter(dictionary.values())
    max_count = max(value_counts.values())
    for key, value in dictionary.items():
        if value_counts[value] == max_count:
            keys_with_max_value = value
    items_with_highest_value = {key: value for key, value in dictionary.items(
    ) if value_counts[value] == max_count}
    return items_with_highest_value, keys_with_max_value


def pipeline(filepath, spkid):
    tmp_folder = f"/tmp/{spkid}"
    os.makedirs(tmp_folder, exist_ok=True)

    # step1 VAD
    data = {"spkid": spkid, "length": 90}
    files = [('file', (filepath, open(filepath, 'rb')))]
    response = send_request(vad_url, files=files, data=data)
    if not response:
        return None

    # step2 截取音频片段
    output_file_li = []
    d = {}
    for idx, i in enumerate(response['timelist']):
        output_file = f"{tmp_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
        extract_audio_segment(filepath, output_file,
                              start_time=i[0]/1000, end_time=i[1]/1000)
        output_file_li.append(output_file)
        d[output_file] = (i[0]/1000, i[1]/1000)

    # step3 普通话过滤
    wav_files = ["local://"+i for i in output_file_li]
    data = {"spkid": spkid, "filelist": ",".join(wav_files)}
    response = send_request(lang_url, data=data)
    if response['code'] == 200:
        pass_list = response['pass_list']
        url_list = response['file_url_list']
        mandarin_wavs = [
            i for i in url_list if pass_list[url_list.index(i)] == 1]
    else:
        logger.error(
            f"Lang_classify failed. spkid:{spkid}.response:{response}")
        return None

    # step4 提取特征
    data = {"spkid": spkid, "filelist": ",".join(mandarin_wavs)}
    response = send_request(encode_url, data=data)
    if response['code'] == 200:
        file_emb = response['file_emb']
    else:
        logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
        return None

    # step5 聚类
    file_emb = file_emb[use_model_type]
    data = {
        "emb_dict": file_emb["embedding"],
        "cluster_line": 3,
        "mer_cos_th": 0.7,
        "cluster_type": "spectral",  # spectral or umap_hdbscan
        "min_cluster_size": 1,
    }
    response = send_request(cluster_url, json=data)
    logger.info(f"\t * -> Cluster result: {response}")
    items, keys_with_max_value = find_items_with_highest_value(
        response['labels'])
    max_score = response['scores'][keys_with_max_value]['max']
    min_score = response['scores'][keys_with_max_value]['min']

    if min_score  < 0.8 :
        logger.info(
            f"After cluster min_score  < 0.8. spkid:{spkid}.response:{response['scores']}")
        return None
    total_duration = 0
    for i in items.keys():
        total_duration += file_emb['length'][i]
    if total_duration < cfg.MIN_LENGTH_REGISTER:
        logger.info(
            f"After cluster total_duration:{total_duration} < {cfg.MIN_LENGTH_REGISTER}s. spkid:{spkid}.response:{response}")
        return None
    selected_files = sorted(items.keys(), key=lambda x: x.split(
        "/")[-1].replace(".wav", "").split("_")[0])
    audio_data = np.concatenate([torchaudio.load(file.replace(
        "local://", ""))[0] for file in selected_files], axis=-1)
    torchaudio.save(os.path.join(tmp_folder, f"{spkid}_selected.wav"), torch.from_numpy(
        audio_data), sample_rate=8000)

    selected_times = [d[_data.replace("local://", "")]
                      for _data in selected_files]

    # step6 ASR
    text = ""
    data = {"spkid": spkid, "postprocess": "1"}
    files = [('wav_file', (filepath, open(filepath, 'rb')))]
    response = send_request(asr_url, files=files, data=data)
    if response.get('transcription') and response.get('transcription').get('text'):
        text = response['transcription']["text"]
        # logger.info(f"\t * -> ASR结果: {text}")
    else:
        logger.error(
            f"ASR failed. spkid:{spkid}.message:{response['message']}")

    # step7 NLP
    # nlp_result = classify_text(text)
    # logger.info(f"\t * -> 文本分类结果: {nlp_result}")

    # step7 话术过滤
    # a, b = check_text(text)
    # if a == "正常":
    #     # todo 查找新话术逻辑
    #     return None

    # step8 上传OSS
    raw_url = upload_file("raw", filepath, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file("raw", os.path.join(
        tmp_folder, f"{spkid}_selected.wav"), f"{spkid}/{spkid}_selected.wav")

    return {
        "spkid": spkid,
        "raw_file_path": raw_url,
        "selected_url": selected_url,
        "asr_result": text,
        "selected_times": selected_times,
        # "nlp_result": nlp_result,
        "total_duration": total_duration,
    }


msg_db = cfg.MYSQL


def insert_to_db(data):
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO check_for_speaker_diraization (`record_id`, `file_url`, `selected_url`, `asr_text`, `wav_duration`,`create_time`,`selected_times`) VALUES (%s, %s, %s, %s, %s,now(), %s);"
        cursor.execute(sql, (data['spkid'], data['raw_file_path'],
                       data['selected_url'], data['asr_result'], data['total_duration'], str(data['selected_times'])))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. spkid:{data['spkid']}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def main(i):
    try:
        spkid = os.path.basename(i).split(".")[0].split('-')[-1]
        tmp_folder = f"/tmp/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)
        pipeline_result = pipeline(i, spkid)
        logger.info(pipeline_result)
        if pipeline_result:
            insert_to_db(pipeline_result)
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. msg:{e}.")
    finally:
        subprocess.call(f"rm -rf {tmp_folder}", shell=True)


if __name__ == "__main__":
    # wav_files = glob.glob("/datasets/changzhou/*.wav")
    wav_files = glob.glob("./*.wav")
    logger.info(f"Total wav files: {len(wav_files)}")
    wav_files = sorted(wav_files)
    process_map(main, wav_files, max_workers=1, desc='TQDMING---:')
