# coding = utf-8
# @Time    : 2023-07-02  11:48:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: nn vad server.

from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

import torch
import cfg

from utils.files import get_sub_wav
from utils.cmd import run_cmd
from utils.preprocess import save_file, save_url
from utils.oss import upload_files
from utils.log import logger, err_logger
app = Flask(__name__)

from vad import lyxx_VAD

def crdnn_vad(filepath, save_folder_path, smooth_threshold=0.5, min_duration=1, save_oss=False):
    os.makedirs(save_folder_path, exist_ok=True)

    try:
        timelist = inference_pipeline(
            audio_in=filepath, audio_fs=16000).get("text")
    except Exception as e:
        return None, None, f"VAD inference failed: {str(e)}",[]
    file_list = get_sub_wav(filepath, save_folder_path,
                            timelist, smooth_threshold, min_duration)

    if save_oss:
        try:
            url_list = upload_files(
                bucket_name="vad", files=file_list, save_days=cfg.MINIO["test_save_days"])
        except Exception as e:
            return None, None, f"File upload to OSS failed: {str(e)}",timelist
    else:
        url_list = []

    run_cmd(f"rm -rf {filepath}")

    return file_list, url_list, None,timelist


@app.route('/crdnn_vad/<filetype>', methods=['POST'])
def main(filetype):
    try:
        # try:
        spkid = request.form.get('spkid')
        channel = int(request.form.get('channel', 0))
        smooth_threshold = float(request.form.get('smooth_threshold', 0.5))
        min_duration = float(request.form.get('min_duration', 2))
        start = request.form.get('start', 0)
        length = request.form.get('length', 999)
        save_oss = request.form.get('save_oss', False)

        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        logger.info(f"* New request: {spkid} ===================================== ")
        logger.info(f"# spkid:{spkid},channel:{channel},smooth_threshold:{smooth_threshold},min_duration:{min_duration},start:{start},length:{length},save_oss:{save_oss}")
        if filetype == "file":
            filedata = request.files.get('file')
            filepath, url = save_file(
                filedata, spkid, channel, upload=save_oss,start=start,length=length,sr=cfg.SR)
            logger.info(f"# filepath:{filepath},url:{url}")
        elif filetype == "url":
            url = request.form.get('url')
            filepath, url = save_url(url, spkid, channel, upload=save_oss,start=start,length=length,sr=cfg.SR)
            logger.info(f"# filepath:{filepath},url:{url}")
        else:
            err_logger.error(f"Invalid filetype: {filetype}")
            return jsonify({"code": 400, "msg": "Invalid filetype"})

        temp_folder = os.path.join(cfg.TEMP_PATH, "nn_vad_server", spkid)
        logger.info(f"# temp_folder:{temp_folder}")
        try:
            file_list, url_list, error_msg,timelist = fsmn_vad(
                filepath, temp_folder, smooth_threshold, min_duration, save_oss)
            logger.info(f"# Result file_list:{file_list},url_list:{url_list},error_msg:{error_msg},timelist:{timelist}"")
        except Exception as e:
            err_logger.error(f"VAD failed: {str(e)}")
            return jsonify({"code": 500, "msg": str(e)})
        if error_msg is not None:
            return jsonify({"code": 500, "msg": error_msg})
        return jsonify({"code": 200, "raw_file_url": url, "msg": "success", "file_list": file_list, "url_list": url_list,"timelist":timelist})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5008)