#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   音频注册单模型接口
'''
import datetime
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
lang_url = f"{host}:5002/lang_classify"  # 语种识别
msg_db = cfg.MYSQL

model_type = "ERES2NET_Base"
BUCKET_NAME = "black"
tmp_folder = "/tmp/register"
os.makedirs(tmp_folder, exist_ok=True)

def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: spkid:{data['spkid']}. msg:{e}.")
        return None


def add_speaker(db_info):
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
    spkid = db_info.get("spkid")
    valid_length = db_info.get("valid_length")
    raw_url = db_info.get("raw_url")
    selected_url = db_info.get("selected_url")
    record_month = db_info.get("record_month")
    asr_text = db_info.get("asr_text")
    record_type = db_info.get("record_type")
    try:
        query_sql = f"insert into black_speaker_info (record_id, valid_length,file_url,preprocessed_file_url,record_month,asr_text,record_type,register_time) VALUES (%s, %s, %s,%s,%s,%s,%s,now())"
        cursor.execute(query_sql, (spkid, valid_length, raw_url, selected_url, record_month,asr_text,record_type))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
        return False
    finally:
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


def compare_handler(model_type=None, embedding=None, black_limit=0.78, top_num=10):
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
    '''
    截取音频片段
    '''
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
        USE_LANG = request.form.get('use_lang', False)  # 是否启动语言检测
        USE_ASR = request.form.get('use_asr', False)    # 是否启动语音识别
        spkid = request.form.get('spkid', "init_id")
        record_month = request.form.get('record_month', datetime.datetime.now().month)
        record_type = request.form.get('record_type', "")
        channel = request.form.get('channel', 0)

        spkid_folder = f"{tmp_folder}/{spkid}"
        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, raw_url = save_file(filedata, spkid, channel=channel, server_name="register")
        else:
            filepath, raw_url = save_url(request.form.get('url'), spkid, channel, server_name="register")

        # VAD
        data = {"spkid": spkid, "length": 90}
        files = [('file', (filepath, open(filepath, 'rb')))]
        response = send_request(vad_url, files=files, data=data)
        if not response:
            logger.error(f"VAD failed. spkid:{spkid}. response:{response}")
            return jsonify({"code": 500, "spkid": spkid, "msg": "VAD failed. response:{}".format(response)})

        # 截取音频片段
        output_file_li = []
        d = {}
        valid_length = 0
        for idx, i in enumerate(response['timelist']):
            output_file = f"{spkid_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
            extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
            output_file_li.append(output_file)
            valid_length += (i[1]-i[0])/1000
            d[output_file] = (i[0]/1000, i[1]/1000)

        if valid_length < 10:
            logger.error(f"VAD failed. spkid:{spkid}. valid_length:{valid_length} .")
            return jsonify({"code": 500, "spkid": spkid, "msg": "VAD failed. valid_length:{}. ".format(valid_length)})
        selected_path = get_joint_wav(spkid, output_file_li)

        if USE_LANG:
            # 普通话过滤
            wav_files = ["local://" + selected_path]
            data = {"spkid": spkid, "filelist": ",".join(wav_files)}
            response = send_request(lang_url, data=data)
            if response['code'] == 200:
                pass_list = response['pass_list']
                url_list = response['file_url_list']
                mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
            else:
                logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
                return jsonify({"code": 500, "spkid": spkid, "msg": "Lang_classify failed. response:{}".format(response)})
        else:
            mandarin_wavs = ["local://" + selected_path]

        # 提取特征
        data = {"spkid": spkid, "filelist": mandarin_wavs}
        response = send_request(encode_url, data=data)
        if response['code'] == 200:
            emb_new = list(response['file_emb'][model_type]["embedding"].values())[0]
            compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
        else:
            logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
            return jsonify({"code": 500, "spkid": spkid, "msg": "Encode failed. response:{}".format(response)})

        if not compare_results['inbase']:
            logger.info(f"Need register. spkid:{spkid}. compare_result:{compare_results}")
            text = ""
            if USE_ASR:
                # ASR
                data = {"spkid": spkid, "postprocess": "1"}
                files = [('wav_file', (filepath, open(filepath, 'rb')))]
                response = send_request(asr_url, files=files, data=data)
                if response.get('transcription') and response.get('transcription').get('text'):
                    text = response['transcription']["text"]
                else:
                    logger.error(f"ASR failed. spkid:{spkid}.message:{response['message']}")

            # upload to oss
            raw_url = upload_file(BUCKET_NAME, filepath, f"{spkid}/raw_{spkid}.wav")
            selected_url = upload_file(BUCKET_NAME, selected_path, f"{spkid}/{spkid}_selected.wav")
            db_info = {}
            db_info['spkid'] = spkid
            db_info['valid_length'] = valid_length
            db_info['raw_url'] = raw_url
            db_info['selected_url'] = selected_url
            db_info['record_month'] = record_month
            db_info['asr_text'] = text
            db_info['record_type'] = record_type
            if add_speaker(db_info):
                to_database(embedding=torch.tensor(emb_new), spkid=spkid, use_model_type=model_type, mode="register")
                logger.info(f"Add speaker success. spkid:{spkid}")
                return jsonify({"code": 200, "spkid": spkid, "msg": "Add speaker success."})
        else:
            logger.info(f"Speaker already exists. spkid:{spkid}. Compare result:{compare_results}")
            return jsonify({"code": 200, "spkid": spkid, "msg": "Speaker already exists. Compare result:{}".format(compare_results)})

    except Exception as e:
        logger.error(f"Register failed. spkid:{spkid}.msg:{e}")
        return jsonify({"code": 500, "spkid": spkid, "msg": "Register failed. msg:{}".format(e)})
    finally:
        shutil.rmtree(spkid_folder)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)
