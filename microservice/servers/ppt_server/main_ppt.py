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
from orm import add_speaker, add_hit, check_phone, get_black_info, get_hit_info, get_hit_spkname,  get_table_info
from tools import compare_handler, extract_audio_segment, find_items_with_highest_value, get_joint_wav, get_similarities_result, pipeline, send_request
from utils.oss.upload import upload_file
from loguru import logger
from utils.orm.db_orm import delete_by_key, get_embeddings, to_database
import os
import cfg
import torch
import pymysql
import numpy as np
from flask import Flask, request, jsonify
from utils.preprocess.save import save_file, save_url
import sys
sys.path.append("/home/xuekaixiang/workplace/vaf/microservice/servers/ppt_server")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

msg_db = cfg.MYSQL


@app.route("/register/<filetype>", methods=["POST"])
def register(filetype):
    """
    register
    """
    tmp_folder = "/tmp/register"
    os.makedirs(tmp_folder, exist_ok=True)
    try:
        phone = request.form.get('phone', "init_id")
        if check_phone(phone):
            logger.info(f"Speaker already exists. spkid:{phone}")
            return jsonify({"code": 200, "phone": phone, "message": "手机号已存在"})
        spkid_folder = f"{tmp_folder}/{phone}"
        channel = request.form.get('channel', 0)
        name = request.form.get('name', "佚名")
        gender = request.form.get('gender', "")
        if filetype == "file":
            filedata = request.files.get('wav_file')
            if not filedata:
                return jsonify({"code": 7001, "phone": phone, "message": "上传音频文件为空"})
            filepath, raw_url = save_file(filedata, phone, start=1.5, channel=channel, server_name="register")
        else:
            filepath, raw_url = save_url(request.form.get('url'), phone, channel, start=1.5, server_name="register")

        # step1 VAD
        data = {"spkid": phone, "length": 90}
        files = [('file', (filepath, open(filepath, 'rb')))]
        response = send_request(cfg.VAD_URL, files=files, data=data)
        if not response:
            logger.error(f"VAD failed. spkid:{phone}. response:{response}")
            return jsonify({"code": 7002, "phone": phone, "message": "VAD请求失败"})

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

        if valid_length < cfg.MIN_LENGTH:
            logger.error(f"VAD failed. spkid:{phone}. valid_length:{valid_length}.file_path:{file_name}")
            return jsonify({"code": 7003, "phone": phone, "message": "音频有效长度小于{}秒，当前{}秒".format(cfg.MIN_LENGTH, valid_length)})

        # 提取特征
        data = {"spkid": phone, "filelist": ["local://"+file_name]}
        response = send_request(cfg.ENCODE_URL, data=data)
        if response['code'] == 200:
            emb_new = list(response['file_emb'][cfg.ENCODE_MODEL_NAME]["embedding"].values())[0]
            # compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
        else:
            logger.error(f"Encode failed. spkid:{phone}.response:{response}")
            return jsonify({"code": 7004, "phone": phone, "message": "编码请求失败"})

        raw_url = upload_file(cfg.BUCKET_NAME, filepath, f"{phone}/raw_{phone}.wav")
        selected_url = upload_file(cfg.BUCKET_NAME, selected_path, f"{phone}/{phone}_selected.wav")

        if cfg.SHOW_PUBLIC:
            raw_url = raw_url.replace(cfg.HOST, cfg.PUBLIC_HOST)
            selected_url = selected_url.replace(cfg.HOST, cfg.PUBLIC_HOST)

        db_info = {}
        db_info['spkid'] = phone
        db_info['name'] = name
        db_info['gender'] = gender
        db_info['raw_url'] = raw_url
        db_info['selected_url'] = selected_url
        db_info['valid_length'] = valid_length
        try:
            info = ph.Phone().find(phone)
            phone_area = info['province'] + "-" + info['city']
        except Exception as e:
            phone_area = ""
        db_info['phone_area'] = phone_area

        if add_speaker(db_info):
            to_database(embedding=torch.tensor(emb_new), spkid=phone, use_model_type=cfg.ENCODE_MODEL_NAME, mode="register")
            logger.info(f"Add speaker success. spkid:{phone}")
            return jsonify({"code": 200, "phone": phone, "message": "注册成功"})
    except Exception as e:
        logger.error(f"Register failed. spkid:{phone}.message:{e}")
        return jsonify({"code": 7007, "phone": phone, "message": "注册失败。错误信息:{}".format(e)})
    # finally:
    #     shutil.rmtree(spkid_folder)


@app.route("/test/<filetype>", methods=["POST"])
def test(filetype):
    try:
        emb_db_dic = {}
        for i in cfg.ENCODE_MODEL_NAME_LIST:
            emb_db_dic[i] = get_embeddings(use_model_type=i)

        spkid = request.form.get('phone', "init_id")
        name = request.form.get('name', "佚名")
        gender = request.form.get('gender', "")
        channel = request.form.get('channel', 0)
        use_cluster = request.form.get('use_cluster', 0)
        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, raw_url = save_file(filedata, spkid, start=1.5, channel=channel, server_name="test")
        else:
            filepath, raw_url = save_url(request.form.get('url'), spkid, start=1.5, channel=channel, server_name="test")

        tmp_folder = f"/tmp/test/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)

        pipeline_result = pipeline(tmp_folder, filepath, spkid, use_cluster)
        if pipeline_result['code'] == 200:
            # 提取特征
            data = {"spkid": spkid, "filelist": "local://"+pipeline_result['selected_path']}
            response = send_request(cfg.ENCODE_URL, data=data)
            if response['code'] == 200:
                file_emb = response['file_emb']
            else:
                logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
                return jsonify({"code": 7004, "message": "编码请求失败"})

            # 撞库
            compare_result = {}
            for i in cfg.ENCODE_MODEL_NAME_LIST:
                hit_spkid, score = get_similarities_result(
                    emb_db_dic, i, np.array(list(emb_db_dic[i].values())),
                    np.array(list(file_emb[i]['embedding'].values())[0]).reshape(1, -1))
                logger.info(f"hit_spkid:{hit_spkid}, score:{score}")
                if score < cfg.COMPARE_SCORE_THRESHOLD:
                    hit_spkname = get_hit_spkname(hit_spkid)
                    # TODO:"model" 改成 i
                    compare_result["model"] = {"is_hit": False, "hit_spkid": hit_spkid, "score": score, "hit_spkname": hit_spkname}
                    logger.info(f"spkid:{spkid} is not in black list. score:{score}")
                    if cfg.SHOW_PUBLIC:
                        del compare_result['model']["score"]
                    return jsonify({"code": 200, "compare_result": compare_result, "message": "{}不在黑库中".format(spkid)})

                # 获取说话人
                hit_spkname = get_hit_spkname(hit_spkid)
                compare_result['model'] = {"is_hit": True, "hit_spkid": hit_spkid, "score": score, "hit_spkname": hit_spkname}

            # OSS
            raw_url = upload_file(cfg.COMPARE_BUCKET_NAME, filepath, f"{spkid}/raw_{spkid}.wav")
            selected_url = upload_file(cfg.COMPARE_BUCKET_NAME, pipeline_result['selected_path'], f"{spkid}/{spkid}_selected.wav")
            if cfg.SHOW_PUBLIC:
                raw_url = raw_url.replace(cfg.HOST, cfg.PUBLIC_HOST)
                selected_url = selected_url.replace(cfg.HOST, cfg.PUBLIC_HOST)

            pipeline_result['name'] = name
            pipeline_result['gender'] = gender
            pipeline_result['raw_url'] = raw_url
            pipeline_result['selected_url'] = selected_url
            pipeline_result['compare_result'] = compare_result
            add_hit(pipeline_result)
            if cfg.SHOW_PUBLIC:
                del compare_result['model']["score"]
            return jsonify({"code": 200, "message": "success", "file_url": raw_url, "compare_result": compare_result})
        else:
            return jsonify(pipeline_result)
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. message:{e}.")
        return jsonify({"code": 500, "message": "比对失败。错误信息{}".format(e)})
    # finally:
    #     if os.path.exists(tmp_folder):
    #         shutil.rmtree(tmp_folder)


@app.route("/get_spkinfo", methods=["POST"])
def get_spk():
    page_no = request.form.get('page_no', "1")
    page_size = request.form.get('page_size', "10")
    ret, total = get_black_info(int(page_no), int(page_size))
    for i in ret:
        i["register_time"] = str(i["register_time"])
    return jsonify({"code": 200, "message": "success", "data": ret, "total": total})


@app.route("/get_hitinfo", methods=["POST"])
def get_hit():
    page_no = request.form.get('page_no', "1")
    page_size = request.form.get('page_size', "10")
    ret, total = get_hit_info(int(page_no), int(page_size))
    for i in ret:
        i["hit_time"] = str(i["hit_time"])
    return jsonify({"code": 200, "message": "success", "data": ret, "total": total})


@app.route("/get_table_info", methods=["GET"])
def get_table():
    ret = get_table_info()
    return jsonify({"code": 200, "message": "success", "data": ret})


@app.route('/update_user', methods=['POST'])
def update_user():
    phone = request.form.get('phone', "")
    new_name = request.form.get('new_name', "")
    if not phone:
        logger.error(f"phone is null")
        return jsonify({"code": 500, "message": "手机号不可以为空"})
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    query_sql = f"update black_speaker_info_ppt set name='{new_name}' where phone='{phone}';"
    cursor.execute(query_sql)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"code": 200, "message": "更新成功"})


@app.route('/get_users', methods=['POST'])
def get_users():
    phone = request.form.get('phone', "")
    if not phone:
        logger.error(f"phone is null")
        return jsonify({"code": 500, "message": "手机号不可以为空"})
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    query_sql = f"select * from black_speaker_info_ppt where phone='{phone}' and status=1;"
    cursor.execute(query_sql)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    for i in result:
        i["register_time"] = str(i["register_time"])
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
    query = f'delete from black_speaker_info_ppt where phone={phone};'
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    delete_by_key(phone)

    return jsonify({'message': 'User deleted successfully', 'code': 200, "phone": phone})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
