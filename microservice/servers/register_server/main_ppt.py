#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   音频注册单模型接口，演示使用，
前提，录制时长符合要求，有效音大于10s 新建一张和speaker一样的表，speaker_ppt
'''
import shutil
from utils.preprocess.save import save_file, save_url
from flask import Flask, request, jsonify
from loguru import logger
from utils.orm.db_orm import get_embeddings, to_database
import wget
from tqdm.contrib.concurrent import process_map
import requests
import os
import cfg
from tqdm import tqdm
import torch
import pymysql
import numpy as np
import time
import multiprocessing
from pydub import AudioSegment
import sys
from utils.oss.upload import upload_file


app = Flask(__name__)
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

host = "http://192.168.3.169"
encode_url = f"{host}:5001/encode"  # 提取特征
cluster_url = f"{host}:5011/cluster"  # cluster
asr_url = f"{host}:5000/transcribe/file"  # ASR
vad_url = f"{host}:5005/energy_vad/file"  # VAD
msg_db = cfg.MYSQL

model_type = "ERES2NET_Base"


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: spkid:{data['spkid']}. msg:{e}.")
        return None


def add_speaker(spkid, raw_url, selected_url):
    """
    录入黑库表
    """
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
        query_sql = f"insert into speaker_ppt (phone, file_url,preprocessed_file_url,register_time) VALUES (%s, %s, %s,now())"
        cursor.execute(query_sql, (spkid, raw_url, selected_url))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def get_selected_url_from_db():
    """
    获取需要注册的音频url
    """
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
        sql = "select record_id,selected_url from check_for_speaker_diraization"
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception as e:
        logger.error(f"Get selected_url from db failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result


def cosine_similarity(input_data):
    base_item, base_embedding, embedding = input_data
    base_embedding = torch.tensor(base_embedding)
    return [similarity(base_embedding, embedding).numpy(), base_item]


def compare_handler(model_type=None, embedding=None, black_limit=0.78,top_num=10):
    """
    是否在黑库中 并返回top1-top10
    """
    emb_db = get_embeddings(use_model_type=model_type)
    embedding = torch.tensor(embedding).to('cpu')
    input_data = [(k, emb_db[k], embedding) for k in emb_db.keys()]

    t1 = time.time()
    results = process_map(cosine_similarity, input_data, max_workers=1, chunksize=1000, desc='Doing----')
    if not results:
        return {'best_score': 0, 'inbase': 0}
    t2 = time.time()
    logger.info(f"compare_handler time:{t2-t1}")

    return_results = {}
    results = sorted(results, key=lambda x: float(x[0]) * (-1))
    return_results["best_score"] = float(np.array(results[0][0]))

    if results[0][0] <= black_limit:
        return_results["inbase"] = 0
        return return_results
    else:
        return_results["inbase"] = 1
        # top1-top10
        if len(results) < top_num:
            top_num = len(results)
        for index in range(top_num):
            return_results[f"top_{index + 1}"] = f"{results[index][0]:.5f}"
            return_results[f"top_{index + 1}_id"] = str(results[index][1])
    return return_results


def extract_audio_segment(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def get_joint_wav(phone, wav_list):
    """
    拼接音频
    :param phone:
    :param wav_list:
    :return:
    """
    playlist = AudioSegment.empty()
    for wav in wav_list:
        playlist = playlist + AudioSegment.from_wav(wav)
    output_name = f'{tmp_folder}/{phone}/{phone}_joint.wav'
    playlist.export(output_name, format='wav')
    return output_name


@app.route("/register/<filetype>", methods=["POST"])
def main(filetype):
    """
    register
    """
    try:
        spkid = request.form.get('spkid', "init_id")
        spkid_folder=f"{tmp_folder}/{spkid}"
        channel = request.form.get('channel', 0)
        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, raw_url = save_file(filedata, spkid, channel=channel, server_name="register")
        else:
            filepath, raw_url = save_url(request.form.get('url'), spkid, channel)

        # step1 VAD
        data = {"spkid": spkid, "length": 90}
        files = [('file', (filepath, open(filepath, 'rb')))]
        response = send_request(vad_url, files=files, data=data)
        if not response:
            return jsonify({"code": 500, "spkid": spkid, "msg": "VAD failed. response:{}".format(response)})

        # step2 截取音频片段
        output_file_li = []
        d = {}
        for idx, i in enumerate(response['timelist']):
            output_file = f"{spkid_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
            extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
            output_file_li.append(output_file)
            d[output_file] = (i[0]/1000, i[1]/1000)

        selected_path = get_joint_wav(spkid, output_file_li)
        file_name = selected_path

        # 提取特征
        data = {"spkid": spkid, "filelist": ["local://"+file_name]}
        response = send_request(encode_url, data=data)
        if response['code'] == 200:
            emb_new = list(response['file_emb'][model_type]["embedding"].values())[0]
            compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
        else:
            logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
            return jsonify({"code": 500, "spkid": spkid, "msg": "Encode failed. response:{}".format(response)})

        if not compare_results['inbase']:
            logger.info(f"Need register. spkid:{spkid}. compare_result:{compare_results}")
            add_success = to_database(embedding=torch.tensor(emb_new), spkid=spkid, use_model_type=model_type, mode="register")
            if add_success:
                # upload to oss
                raw_url = upload_file("raw", filepath, f"{spkid}/raw_{spkid}.wav")
                selected_url = upload_file("raw", selected_path, f"{spkid}/{spkid}_selected.wav")

                logger.info(f"Add speaker success. spkid:{spkid}")
                add_speaker(spkid, raw_url, selected_url)
                return jsonify({"code": 200, "spkid": spkid, "msg": "Add speaker success."})
        else:
            logger.info(f"Speaker already exists. spkid:{spkid}. Compare result:{compare_results}")
            return jsonify({"code": 200, "spkid": spkid, "msg": "Speaker already exists. Compare result:{}".format(compare_results)})

    except Exception as e:
        logger.error(f"Register failed. spkid:{spkid}.msg:{e}")
    finally:
        shutil.rmtree(spkid_folder)

if __name__ == "__main__":
    tmp_folder = "/tmp/register"
    os.makedirs(tmp_folder, exist_ok=True)
    app.run(host="0.0.0.0", port=8899)
