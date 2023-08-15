#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   音频推理，演示用
'''
import phone as ph
from orm import add_speaker, add_hit, get_black_info, get_hit_info, get_table_info
from tools import compare_handler, extract_audio_segment, find_items_with_highest_value, get_joint_wav, get_similarities_result, send_request
from collections import Counter
from utils.oss.upload import upload_file
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from loguru import logger
from utils.orm.db_orm import get_embeddings, to_database
import wget
from tqdm.contrib.concurrent import process_map
import requests
import os
import cfg
from tqdm import tqdm
import torchaudio
import torch
import pymysql
import numpy as np
import time
import glob
import shutil
from flask import Flask, request, jsonify
from utils.preprocess.save import save_file, save_url
import sys
sys.path.append("/home/xuekaixiang/workplace/vaf/microservice/servers/ppt_server")


app = Flask(__name__)


name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

host = "http://192.168.3.169"
encode_url = f"{host}:5001/encode"  # 提取特征
cluster_url = f"{host}:5011/cluster"  # cluster
asr_url = f"{host}:5000/transcribe/file"  # ASR
vad_url = f"{host}:5005/energy_vad/file"  # VAD
lang_url = f"{host}:5002/lang_classify"  # 语种识别
model_type = "ERES2NET_Base"
ENCODE_MODEL_LIST = ["ERES2NET_Base"]


@app.route("/register/<filetype>", methods=["POST"])
def register(filetype):
    """
    register
    """
    tmp_folder = "/tmp/register"
    os.makedirs(tmp_folder, exist_ok=True)
    try:
        phone = request.form.get('phone', "init_id")
        spkid_folder = f"{tmp_folder}/{phone}"
        channel = request.form.get('channel', 0)
        name = request.form.get('name', "佚名")
        gender = request.form.get('gender', "")
        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, raw_url = save_file(filedata, phone, channel=channel, server_name="register")
        else:
            filepath, raw_url = save_url(request.form.get('url'), phone, channel, server_name="register")

        # step1 VAD
        data = {"spkid": phone, "length": 90}
        files = [('file', (filepath, open(filepath, 'rb')))]
        response = send_request(vad_url, files=files, data=data)
        if not response:
            logger.error(f"VAD failed. spkid:{phone}.response:{response}") 
            return jsonify({"code": 500, "phone": phone, "msg": "VAD failed. response:{}".format(response)})

        # step2 截取音频片段
        output_file_li = []
        d = {}
        valid_length = 0
        for idx, i in enumerate(response['timelist']):
            output_file = f"{spkid_folder}/{phone}_{idx}.wav"  # 截取后的音频片段保存路径
            extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
            output_file_li.append(output_file)
            valid_length += (i[1]-i[0])/1000
            d[output_file] = (i[0]/1000, i[1]/1000)


        selected_path = get_joint_wav(tmp_folder, phone, output_file_li)
        file_name = selected_path

        if valid_length < 10:
            logger.error(f"VAD failed. spkid:{phone}. valid_length:{valid_length}.file_path:{file_name}")
            return jsonify({"code": 500, "phone": phone, "msg": "VAD failed. valid_length:{}".format(valid_length)})
        
        # 提取特征
        data = {"spkid": phone, "filelist": ["local://"+file_name]}
        response = send_request(encode_url, data=data)
        if response['code'] == 200:
            emb_new = list(response['file_emb'][model_type]["embedding"].values())[0]
            compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
        else:
            logger.error(f"Encode failed. spkid:{phone}.response:{response}")
            return jsonify({"code": 500, "phone": phone, "msg": "Encode failed. response:{}".format(response)})

        if not compare_results['inbase']:
            logger.info(f"Need register. spkid:{phone}. compare_result:{compare_results}")
            add_success = to_database(embedding=torch.tensor(emb_new), spkid=phone, use_model_type=model_type, mode="register")
            if add_success:
                # upload to oss
                raw_url = upload_file("raw", filepath, f"{phone}/raw_{phone}.wav")
                selected_url = upload_file("raw", selected_path, f"{phone}/{phone}_selected.wav")
                logger.info(f"Add speaker success. spkid:{phone}")

                db_info = {}
                db_info['spkid'] = phone
                db_info['name'] = name
                db_info['gender'] = gender
                db_info['raw_url'] = raw_url
                db_info['selected_url'] = selected_url
                db_info['valid_length'] = valid_length
                info = ph.Phone().find(phone)
                phone_area = info['province'] +"-"+ info['city']
                db_info['phone_area'] = phone_area

                add_speaker(db_info)
                logger.info(f"Add speaker success. spkid:{phone}")
                return jsonify({"code": 200, "phone": phone, "msg": "Add speaker success."})
        else:
            logger.info(f"Speaker already exists. spkid:{phone}. Compare result:{compare_results}")
            ret_dic = {
                "code": 200,
                "phone": phone,
                "inbase": compare_results['inbase'],
                "hit_spkid": compare_results['top_1_id'],
                "best_score": compare_results['best_score'],
                "msg": "Speaker already exists. Compare result:{}".format(compare_results)
            }
            return jsonify(ret_dic)

    except Exception as e:
        logger.error(f"Register failed. spkid:{phone}.msg:{e}")
    finally:
        shutil.rmtree(spkid_folder)


def pipeline(tmp_folder, filepath, spkid,use_cluster):
    # step1 VAD
    data = {"spkid": spkid, "length": 90}
    files = [('file', (filepath, open(filepath, 'rb')))]
    response = send_request(vad_url, files=files, data=data)
    if not response:
        logger.error(f"VAD failed. spkid:{spkid}.response:{response}")
        return {"code": 500, "msg": "VAD failed."}

    # step2 截取音频片段
    output_file_li = []
    d = {}
    valid_length = 0
    for idx, i in enumerate(response['timelist']):
        output_file = f"{tmp_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
        extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
        output_file_li.append(output_file)
        valid_length += (i[1]-i[0])/1000
        d[output_file] = (i[0]/1000, i[1]/1000)

    if valid_length < 5:
        logger.error(f"VAD failed. spkid:{spkid}. valid_length:{valid_length}.output_file:{output_file}")
        return {"code": 500, "phone": spkid, "msg": "VAD failed. valid_length:{}. output_file:{}".format(valid_length,output_file)}

    # step3 普通话过滤
    wav_files = ["local://"+i for i in output_file_li]
    data = {"spkid": spkid, "filelist": ",".join(wav_files)}
    response = send_request(lang_url, data=data)
    if response['code'] == 200:
        pass_list = response['pass_list']
        url_list = response['file_url_list']
        mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
    else:
        logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
        return {"code": 500, "msg": "Lang_classify failed."}

    # step4 提取特征
    data = {"spkid": spkid, "filelist": ",".join(mandarin_wavs)}
    response = send_request(encode_url, data=data)
    if response['code'] == 200:
        file_emb = response['file_emb']
    else:
        logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
        return {"code": 500, "msg": "Encode failed."}

    if use_cluster:
        # step5 聚类
        file_emb = file_emb[model_type]
        data = {
            "emb_dict": file_emb["embedding"],
            "cluster_line": 3,
            "mer_cos_th": 0.7,
            "cluster_type": "spectral",  # spectral or umap_hdbscan
            "min_cluster_size": 1,
        }
        response = send_request(cluster_url, json=data)
        logger.info(f"\t * -> Cluster result: {response}")
        items, keys_with_max_value = find_items_with_highest_value(response['labels'])
        max_score = response['scores'][keys_with_max_value]['max']
        min_score = response['scores'][keys_with_max_value]['min']

        if min_score < 0.8:
            logger.info(f"After cluster min_score  < 0.8. spkid:{spkid}.response:{response['scores']}")
            return {"code": 500, "msg": "After cluster min_score  < 0.8."}
        selected_files = sorted(items.keys(), key=lambda x: x.split("/")[-1].replace(".wav", "").split("_")[0])
    else:
        selected_files=response['file_url_list']
    
    audio_data = np.concatenate([torchaudio.load(file.replace("local://", ""))[0] for file in selected_files], axis=-1)
    _selected_path = os.path.join(tmp_folder, f"{spkid}_selected.wav")
    torchaudio.save(_selected_path, torch.from_numpy(audio_data), sample_rate=8000)

    selected_times = [d[_data.replace("local://", "")] for _data in selected_files]

    return {
        "code": 200,
        "spkid": spkid,
        "raw_file_path": filepath,
        "selected_path": _selected_path,
        "selected_times": selected_times,
        "total_duration": valid_length,
    }


@app.route("/test/<filetype>", methods=["POST"])
def test(filetype):
    try:
        emb_db_dic = {}
        for i in ENCODE_MODEL_LIST:
            emb_db_dic[i] = get_embeddings(use_model_type=i)
        spkid = request.form.get('phone', "init_id")
        name = request.form.get('name', "佚名")
        gender = request.form.get('gender', "")
        channel = request.form.get('channel', 0)
        use_cluster = request.form.get('use_cluster', 0)
        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, raw_url = save_file(filedata, spkid, sr=8000, channel=channel, server_name="test")
        else:
            filepath, raw_url = save_url(request.form.get('url'), spkid, sr=8000, channel=channel, server_name="test")

        tmp_folder = f"/tmp/test/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)
        pipeline_result = pipeline(tmp_folder, filepath, spkid,use_cluster)
        if pipeline_result['code'] == 200:
            # 提取特征
            data = {"spkid": spkid, "filelist": "local://"+pipeline_result['selected_path']}
            response = send_request(encode_url, data=data)
            if response['code'] == 200:
                file_emb = response['file_emb']
            else:
                logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
                return jsonify({"code": 500, "message": "Encode failed."})

            # 撞库
            compare_result = {}
            for i in cfg.ENCODE_MODEL_LIST:
                hit_spkid, score = get_similarities_result(
                    emb_db_dic, i, np.array(list(emb_db_dic[i].values())),
                    np.array(list(file_emb[i]['embedding'].values())[0]).reshape(1, -1))
                logger.info(f"hit_spkid:{hit_spkid}, score:{score}")
                if score < 0.6:
                    logger.info(f"spkid:{spkid} is not in black list. score:{score}")
                    return jsonify({"code": 200, "message": "{} is not in black list. hit_spkid:{}, score:{}.".format(spkid, hit_spkid, score)})
                compare_result[i] = {"is_hit": True, "hit_spkid": hit_spkid, "score": score}

            # OSS
            raw_url = upload_file("test", filepath, f"{spkid}/raw_{spkid}.wav")
            selected_url = upload_file("test", pipeline_result['selected_path'], f"{spkid}/{spkid}_selected.wav")
            pipeline_result['name'] = name
            pipeline_result['gender'] = gender
            pipeline_result['raw_url'] = raw_url
            pipeline_result['selected_url'] = selected_url
            pipeline_result['compare_result'] = compare_result
            add_hit(pipeline_result)
            return jsonify({"code": 200, "message": "success", "file_url": raw_url, "compare_result": compare_result})
        else:
            return jsonify(pipeline_result)
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. msg:{e}.")
        return jsonify({"code": 500, "message": "{}".format(e)})
    # finally:
    #     if os.path.exists(tmp_folder):
    #         shutil.rmtree(tmp_folder)


@app.route("/get_spkinfo", methods=["POST"])
def get_spk():
    page_no = request.form.get('page_no', "1")
    page_size = request.form.get('page_size', "10")
    ret,total = get_black_info(int(page_no), int(page_size))
    for i in ret:
        i["register_time"]= str(i["register_time"]) 
    return jsonify({"code": 200, "message": "success", "data": ret,"total":total})

@app.route("/get_hitinfo", methods=["POST"])
def get_hit():
    page_no = request.form.get('page_no', "1")
    page_size = request.form.get('page_size', "10")
    ret,total= get_hit_info(int(page_no), int(page_size))
    for i in ret:
        i["hit_time"]= str(i["hit_time"]) 
    return jsonify({"code": 200, "message": "success", "data": ret,"total":total})

@app.route("/get_table_info", methods=["GET"])
def get_table():
    ret = get_table_info()
    return jsonify({"code": 200, "message": "success", "data": ret})

msg_db = cfg.MYSQL

@app.route('/get_users', methods=['POST'])
def get_users():
    phone = request.form.get('phone', "init_id")
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    query_sql = f"select * from black_speaker_info_ppt where phone={phone};"
    cursor.execute(query_sql)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return jsonify({"code": 200, "message": "success", "data": result})


@app.route('/delete_user', methods=['POST'])
def delete_user():
    phone = request.form.get('phone')
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    query = f'DELETE FROM black_speaker_info_ppt WHERE phone ={phone} '
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'User deleted successfully', 'code': 200, "phone": phone})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8989)
