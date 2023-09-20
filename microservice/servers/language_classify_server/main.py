
# coding = utf-8
# @Time    : 2023-07-02  11:48:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: nn vad server.

import os
from flask import Flask, request, jsonify
from loguru import logger
import cfg
from utils.cmd import run_cmd, remove_father_path
from utils.preprocess import save_file, save_url
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

language_id = EncoderClassifier.from_hparams(
    source="models/LANG", savedir=f"./pretrained_models/lang-id-ecapa", run_opts={"device": cfg.LANGUAGE_DEVICE})
language_id.eval()
logger.info("Load language id model success.")
logger.info(f"Model device: {cfg.LANGUAGE_DEVICE}\nModel Path: ./pretrained_models/lang-id-ecapa")
app = Flask(__name__)


def mandarin_filter(filelist, score_threshold=0.7):
    pass_list = []
    data_list = []
    message = ""
    for filepath in filelist:
        wavdata = torchaudio.load(filepath)[0].reshape(-1)
        data_list.append(wavdata)
    wavdata = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True).to(cfg.LANGUAGE_DEVICE)
    result = language_id.classify_batch(wavdata)
    for _index in range(len(filelist)):
        if result[1][_index].exp() > score_threshold and result[3][_index].startswith("zh"):
            pass_list.append(1)
            message += f"#Pass file: {filelist[_index]} is {result[3][_index]},  {result[1][_index]} {result[2][_index]}\n"
        else:
            pass_list.append(0)
            message += f"#Error file: {filelist[_index]} is {result[3][_index]},  {result[1][_index]} {result[2][_index]}\n"
    return pass_list, message


@app.route('/lang_classify', methods=['POST'])
def main():
    try:
        # try:
        spkid = request.form.get('spkid')
        channel = int(request.form.get('channel', 0))
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 999))
        end = start + length
        filelist = request.form.get('filelist').split(",")
        save_oss = request.form.get('save_oss', False)
        score_threshold = float(request.form.get('score_threshold', 0.7))

        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        logger.info(f"* New request: {spkid} ===================================== ")
        logger.info(
            f"# spkid: {spkid} channel: {channel} start: {start} length: {length} save_oss: {save_oss} filelist: {filelist} score_threshold: {score_threshold}")

        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True

        local_file_list = []
        file_url_list = []

        for fileurl in filelist:
            try:
                filepath, url = save_url(fileurl, spkid, channel, upload=save_oss, start=start, length=length, sr=cfg.SR, server_name="lang")
                local_file_list.append(filepath)
                file_url_list.append(url)
                logger.info(f"URL-><{fileurl}> File download success: {filepath},{url}")
            except Exception as e:
                logger.error(f"File download failed: {str(e)}")
                return jsonify({"code": 500, "msg": f"File download failed: {str(e)}"})

        pass_list, message = mandarin_filter(local_file_list, score_threshold)
        remove_father_path(filepath)
        torch.cuda.empty_cache()
        return jsonify({"code": 200, "msg": message, "pass_list": pass_list, "file_url_list": file_url_list})
    except Exception as e:
        # remove_father_path(filepath)
        torch.cuda.empty_cache()
        return jsonify({"code": 500, "msg": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
