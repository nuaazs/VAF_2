#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   音频推理，演示用
'''
import traceback
from tools.orm_handler import delete_by_key
from orm import add_speaker, add_hit, check_phone, get_black_info, get_hit_info, get_hit_spkname,  get_table_info
from loguru import logger
import os
import cfg
import pymysql
from flask import Flask, request, jsonify
import sys
from pipeline.register import register_pipeline
from pipeline.search import search_pipeline

sys.path.append("/home/xuekaixiang/workplace/vaf/microservice/servers/ppt_server")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

msg_db = cfg.MYSQL


@app.errorhandler(500)
def internal_server_error(e):
    logger.error(traceback.format_exc())
    return jsonify({"code": 500, "message": "服务器内部错误"})


@app.route("/register/<filetype>", methods=["POST"])
def register(filetype):
    data_info_dict = register_pipeline(request, filetype)
    return jsonify(data_info_dict)


@app.route("/test/<filetype>", methods=["POST"])
def search(filetype):
    """
    search
    """
    data_info_dict = search_pipeline(request, filetype)
    return jsonify(data_info_dict)


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
    app.run(host="0.0.0.0", port=8989)
