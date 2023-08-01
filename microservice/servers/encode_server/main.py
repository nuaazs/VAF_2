# utils


# coding = utf-8
# @Time    : 2023-07-02  11:48:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: encode server

from flask import Flask, request, jsonify
import os
import cfg
import torch
from utils.files import get_sub_wav
from utils.cmd import run_cmd,remove_father_path
from utils.preprocess import save_file, save_url
from utils.oss import upload_files
from utils.log import logger, err_logger

import importlib
import torchaudio
import numpy as np

ENCODE_MODEL_LIST = cfg.ENCODE_MODEL_LIST


emb_dict = {}
for model in ENCODE_MODEL_LIST:
    module = importlib.import_module(f"{model}")
    emb_dict[model] = module.emb
    logger.info(f"-> Load Model from {model}")
    print(f"From {model} load model {emb_dict[model]} successfully.")
    # logger.info(f"-> Model device: {cfg.ENCODE_DEVICE}")

app = Flask(__name__)


def encode_files(wav_files,raw_file_list,start,end):
    file_emb = {}
    message = ""
    for model in cfg.ENCODE_MODEL_LIST:
        file_emb[model] = {}
        file_emb[model]["embedding"] = {}
        file_emb[model]["length"] = {}

        i = 0
        for _index,wav_file in enumerate(wav_files):
            _data,sr = torchaudio.load(wav_file)
            assert sr == cfg.SR, f"File {wav_file} <{raw_file_list[_index]}>  sr is {sr}, not {cfg.SR}."
            _data = _data.reshape(1, -1)
            _data = _data[:, start*sr:end*sr]
            if _data.shape[1] < cfg.SR * 0.1:
                message += f"File {wav_file} <{raw_file_list[_index]}> is too short, only {_data.shape[1]}.\n"
            _data = _data.to(cfg.ENCODE_CAMPP_DEVICE)
            print(f"{emb_dict[model]} encode batch {_index}")
            embeddings = emb_dict[model].encode_batch(_data)
            embeddings = embeddings.detach().cpu().numpy().reshape(-1)
            embeddings = embeddings.astype(np.float32).reshape(-1).tolist()
            file_emb[model]["embedding"][raw_file_list[_index]] = embeddings
            file_emb[model]["length"][raw_file_list[_index]] = _data.shape[1] / cfg.SR
            i += 1
    return file_emb, message


@app.route('/encode', methods=['POST'])
def main():
    try:
        spkid = request.form.get('spkid')
        channel = int(request.form.get('channel', 0))
        filelist = request.form.get('filelist').split(",")
        save_oss = request.form.get('save_oss', False)
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 999))
        end = start + length
        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        # score_threshold = float(request.form.get('score_threshold'))
        local_file_list = []
        raw_file_list = []
        file_url_list = []
        logger.info(f"* New request: {spkid} ===================================== ")
        logger.info(f"# Payload: {request.form}")

        for fileurl in filelist:
            logger.info(f"-> File url: {fileurl}")
            filepath, url = save_url(fileurl.strip(), spkid, channel, upload=save_oss,start=start,length=length,server_name="encode")
            logger.info(f"\t Result -> File path: {filepath}")
            logger.info(f"\t Result -> Url: {url}")
            local_file_list.append(filepath)
            file_url_list.append(url)
            raw_file_list.append(fileurl)

        file_emb, message = encode_files(local_file_list,raw_file_list,start,end)
        # change file_emb to list
        # empty cuda cache
        remove_father_path(filepath)
        torch.cuda.empty_cache()
        return jsonify({"code": 200, "msg": message, "file_emb": file_emb, "file_url_list": file_url_list})
    except Exception as e:
        torch.cuda.empty_cache()
        # remove_father_path(filepath)
        return jsonify({"code": 500, "msg": str(e)})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
