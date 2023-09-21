
# coding = utf-8
# @Time    : 2023-07-02  11:48:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: nn vad server.

from flask import Flask, request, jsonify
import os
from loguru import logger
import cfg
from utils.cmd import run_cmd, remove_father_path
from utils.preprocess import save_file, save_url
from speechbrain.pretrained import EncoderClassifier
import torch
import cfg
import numpy as np

import random


name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

gender_id = EncoderClassifier.from_hparams(
    source="models/gender", savedir=f"./pretrained_models/gender-id-ecapa", run_opts={"device": cfg.GENDER_DEVICE})
gender_id.eval()
logger.info(f"Load Model from models/gender")
logger.info(f"Model device: {cfg.GENDER_DEVICE}")
app = Flask(__name__)


def mandarin_filter(filelist):
    result = {}
    for files in filelist:
        result[files] = {}
        result[files]["male_score"] = {}
        result[files]["female_score"] = {}
        result[files]["gender"] = {}
        out_prob, score, index, text_lab = gender_id.classify_file(files)
        out_prob = out_prob.cpu().numpy().tolist()
        score1 = np.exp(out_prob[0][0])
        score2 = np.exp(out_prob[0][1])
        total = score1 + score2
        norm_score1 = score1 / total
        norm_score2 = score2 / total
        result[files]["gender"] = text_lab[0]
        result[files]["male_score"] = norm_score1
        result[files]["female_score"] = norm_score2
    return result


@app.route('/gender_classify', methods=['POST'])
def main():
    spkid = request.form.get('spkid')
    temp_folder = os.path.join(cfg.TEMP_PATH, "gender", spkid)
    try:

        channel = int(request.form.get('channel', 0))
        filelist = request.form.get('filelist').split(",")
        save_oss = request.form.get('save_oss', False)
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 999))
        end = start + length
        logger.info(f"* New request: {spkid} ===================================== ")
        logger.info(f"# Payload: {request.form}")
        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        local_file_list = []
        file_url_list = []
        print(filelist)
        uuids = []
        for fileurl in filelist:
            # random generate uuid
            # 随机生成uuid
            uuid = str(random.randint(10000000, 99999999))
            uuids.append(uuid)
            logger.info(f"# uuid: {uuid}")
            print(uuid)

            filepath, url = save_url(fileurl, spkid, channel, upload=save_oss, start=start, length=length, sr=cfg.SR, uuid=uuid, server_name="gender")
            local_file_list.append(filepath)
            file_url_list.append(url)

        result = mandarin_filter(local_file_list)
        torch.cuda.empty_cache()
        for uuid in uuids:
            run_cmd(f'rm raw_{uuid}.wav')
        remove_father_path(filepath)
        return jsonify({"code": 200,  "Gender_result": result, "file_url_list": file_url_list})
    except Exception as e:
        torch.cuda.empty_cache()
        # for uuid in uuids:
        #     run_cmd(f'rm raw_{uuid}.wav')
        # remove_father_path(filepath)
        run_cmd(f"rm -rf {temp_folder}")
        return jsonify({"code": 500, "msg": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5009)
