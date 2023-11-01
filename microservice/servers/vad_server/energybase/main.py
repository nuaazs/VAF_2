# coding = utf-8
# @Time    : 2023-07-02  11:48:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: nn vad server.

from flask import Flask, request, jsonify
import os

from loguru import logger
import cfg
from utils.cmd import run_cmd
from utils.preprocess import save_file, save_url

app = Flask(__name__)

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)


def energybase_vad(filepath, save_folder_path, smooth_threshold=0.5, min_duration=2, energy_thresh=1e8, save_oss=False, split=False):
    os.makedirs(save_folder_path, exist_ok=True)
    bin_path = f"{filepath.split('/')[-1][:-4]}.bin"
    bin_path = os.path.join(save_folder_path, bin_path)
    run_cmd(f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  {bin_path}", util_exist=bin_path)

    # cout = run_cmd(f"./vad {bin_path} {save_folder_path}/output.txt {smooth_threshold} {min_duration}", check=False)
    cout = run_cmd(
        f'./vad --wav-bin={bin_path} --energy-thresh={energy_thresh} --text-out={save_folder_path}/output.txt --smooth-threshold={smooth_threshold} --min-duration={min_duration}', check=False)
    logger.info(f"VAD output: {cout}")
    if split:
        # TODO split
        pass
        # file_list = [os.path.join(save_folder_path, _file) for _file in os.listdir(
        #     save_folder_path) if _file.endswith(".wav") and "raw" not in _file]
        # timelist =  []
        # for _file in file_list:
        #     start = float(_file.split("/")[-1].replace(".wav","").split("_")[0])
        #     end = float(_file.split("/")[-1].replace(".wav","").split("_")[1])
        #     timelist.append([start*1000,end*1000])
        # run_cmd(f"rm -rf {bin_path}")
        # if save_oss:
        #     try:
        #         url_list = upload_files(
        #             bucket_name="vad", files=file_list, save_days=cfg.MINIO["test_save_days"])
        #     except Exception as e:
        #         return None, None, f"File upload to OSS failed: {str(e)}",timelist
        # else:
        #     url_list = []
    else:
        # timelist read from {save_folder_path}/output.txt
        timelist = []
        file_list = []
        url_list = []
        with open(f"{save_folder_path}/output.txt", "r") as f:
            for line in f.readlines():
                start, end = line.strip().split(",")
                timelist.append([float(start)*1000, float(end)*1000])
    return file_list, url_list, None, timelist


@app.route('/energy_vad/<filetype>', methods=['POST'])
def main(filetype):
    # try:

    spkid = request.form.get('spkid')
    temp_folder = os.path.join(cfg.TEMP_PATH, "nn_vad_server", spkid)
    channel = int(request.form.get('channel', 0))
    smooth_threshold = float(request.form.get('smooth_threshold', 0.5))
    min_duration = float(request.form.get('min_duration', 2))
    energy_thresh = float(request.form.get('energy_thresh', 1e8))
    start = int(request.form.get('start', 0))
    length = int(request.form.get('length', 999))
    end = start + length
    save_oss = request.form.get('save_oss', False)
    split = request.form.get('split', False)
    url = request.form.get('url', None)
    file = request.files.get('file', None)

    if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
        save_oss = True
    logger.info(f"* New request: {spkid} ===================================== ")
    logger.info(f"# spkid:{spkid},channel:{channel},smooth_threshold:{smooth_threshold},min_duration:{min_duration},energy_thresh:{energy_thresh},start:{start},length:{length},save_oss:{save_oss},file:{file},url:{url}")
    if filetype == "file":
        filedata = request.files.get('file')
        filepath, url = save_file(filedata, spkid, channel, upload=save_oss, start=start, length=length, sr=cfg.SR, server_name="nn_vad_server")
        logger.info(f"# filepath:{filepath},url:{url}")
    elif filetype == "url":
        url = request.form.get('url')
        filepath, url = save_url(url, spkid, channel, upload=save_oss, start=start, length=length, sr=cfg.SR, server_name="nn_vad_server")
        logger.info(f"# filepath:{filepath},url:{url}")
    else:
        logger.error(f"Invalid filetype: {filetype}")
        return jsonify({"code": 400, "msg": "Invalid filetype"})

    logger.info(f"# temp_folder:{temp_folder}")
    try:
        file_list, url_list, error_msg, timelist = energybase_vad(
            filepath, temp_folder, smooth_threshold, min_duration, energy_thresh, save_oss, split)
        logger.info(f"# Result file_list:{file_list},url_list:{url_list},error_msg:{error_msg},timelist:{timelist}")

    except Exception as e:
        logger.error(f"VAD failed: {str(e)}")
        # run_cmd(f"rm -rf {temp_folder}")
        # run_cmd(f"rm -rf {filepath}")
        return jsonify({"code": 500, "msg": str(e)})
    if error_msg is not None:
        # run_cmd(f"rm -rf {temp_folder}")
        # run_cmd(f"rm -rf {filepath}")
        return jsonify({"code": 500, "msg": error_msg})

    # torch.cuda.empty_cache()
    # run_cmd(f"rm -rf {temp_folder}")
    # run_cmd(f"rm -rf {filepath}")
    # remove folder by os
    os.system(f"rm -rf {temp_folder}")
    os.system(f"rm -rf {filepath}")
    return jsonify({"code": 200, "raw_file_url": url, "msg": "success", "file_list": file_list, "url_list": url_list, "timelist": timelist})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
