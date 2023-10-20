#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   
'''
from flask import Flask, request, jsonify
from loguru import logger
import os
from pipeline.register import register_pipeline
from pipeline.encode_pipeline import encode_pipeline
from pipeline.vad_pipeline import vad_pipeline
from pipeline.search_pipeline import search_pipeline
import traceback

app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False
name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)


@app.errorhandler(500)
def internal_server_error(e):
    logger.error(traceback.format_exc())
    return jsonify({"code": 500, "message": "服务器内部错误"})

# 添加一个health check接口
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"code": 200, "message": "ok"})

@app.route("/register/<filetype>", methods=["POST"])
def register(filetype):
    data_info_dict = register_pipeline(request, filetype)
    return jsonify(data_info_dict)


@app.route('/encode/<filetype>', methods=['POST'])
def encode(filetype):
    data_info_dict = encode_pipeline(request, filetype)
    return jsonify(data_info_dict)


@app.route('/energy_vad/<filetype>', methods=['POST'])
def vad(filetype):
    data_info_dict = vad_pipeline(request, filetype)
    return jsonify(data_info_dict)


@app.route('/search/<filetype>', methods=['POST'])
def search(filetype):
    data_info_dict = search_pipeline(request, filetype)
    return jsonify(data_info_dict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5550)
