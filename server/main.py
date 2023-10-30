#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/10/09 10:25:44
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

import os
import random
import shutil
import time
from flask import Flask, request, jsonify
from loguru import logger
import cfg
import torch
from utils import get_embeddings_for_spkid, get_spkid, run_cmd, save_file, to_database

import torchaudio
import numpy as np
from dguard.interface.pretrained import load_by_name

app = Flask(__name__)
random.seed(123)

logger.add("log/{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

model_info = {}
for model_type in cfg.ENCODE_MODEL_LIST:
    model_info_dict ={}
    model_name = model_type
    device = cfg.DEVICE
    model, feature_extractor, sample_rate = load_by_name(model_name, device)
    model_info_dict["model"] = model
    model_info_dict["feature_extractor"] = feature_extractor
    model_info_dict["sample_rate"] = sample_rate
    model.eval()
    model.to(device)
    model_info[model_type] = model_info_dict


def encode_files(wav_files, raw_file_list, start=0, end=999, need_list=False):
    file_emb = {}
    for model_type in cfg.ENCODE_MODEL_LIST:
        model_info_dict = model_info[model_type]
        model = model_info_dict["model"]
        feature_extractor = model_info_dict["feature_extractor"]
        sample_rate = model_info_dict["sample_rate"]
        
        file_emb[model_type] = {}
        file_emb[model_type]["embedding"] = {}

        for _index, wav_file in enumerate(wav_files):
            _data, sr = torchaudio.load(wav_file)
            assert sr == sample_rate, f"File {wav_file} <{raw_file_list[_index]}>  sr is {sr}, not {sample_rate}."
            _data = _data.reshape(1, -1)
            _data = _data[:, int(start*sr):int(end*sr)]
            _data_a = _data[:, :int(_data.shape[1]/2)]
            _data_b = _data[:, int(_data.shape[1]/2):]

            feat = feature_extractor(_data)
            feat = feat.unsqueeze(0)
            feat = feat.to(device)
            with torch.no_grad():
                embeddings = model(feat)[-1].detach().cpu().numpy()
            embeddings = embeddings.astype(np.float32).reshape(1,-1)

            feat_a = feature_extractor(_data_a)
            feat_a = feat_a.unsqueeze(0)
            feat_a = feat_a.to(device)
            with torch.no_grad():
                embeddings_a = model(feat_a)[-1].detach().cpu().numpy()
            embeddings_a = embeddings_a.astype(np.float32).reshape(1,-1)

            feat_b = feature_extractor(_data_b)
            feat_b = feat_b.unsqueeze(0)
            feat_b = feat_b.to(device)
            with torch.no_grad():
                embeddings_b = model(feat_b)[-1].detach().cpu().numpy()
            embeddings_b = embeddings_b.astype(np.float32).reshape(1,-1)

            # concat embeddings_a, embeddings_b, embeddings
            embeddings = np.concatenate((embeddings_a, embeddings_b, embeddings), axis=0).reshape(-1)
            if need_list:
                file_emb[model_type]["embedding"][raw_file_list[_index]] = embeddings.tolist()
            else:
                file_emb[model_type]["embedding"][raw_file_list[_index]] = embeddings
    return file_emb


def energybase_vad(filepath, save_folder_path, smooth_threshold=0.5, min_duration=0.25, energy_thresh=1e8):
    os.makedirs(save_folder_path, exist_ok=True)
    bin_path = filepath.replace(os.path.basename(filepath).split('.')[-1], "bin")
    bin_path = os.path.join(save_folder_path, bin_path)
    # TODO: check if bin_path exist
    run_cmd(f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  {bin_path}", util_exist=bin_path)

    vad_output_path = f"{save_folder_path}/vad_{os.path.basename(filepath)}"
    cout = run_cmd(
        f'./vad_wav --wav-bin={bin_path} --energy-thresh={energy_thresh} --text-out={save_folder_path}/output.txt --smooth-threshold={smooth_threshold} --min-duration={min_duration} --wav-out={vad_output_path}', check=False)
    voice_length = 0
    with open(f"{save_folder_path}/output.txt", "r") as f:
        for line in f.readlines():
            start, end = line.strip().split(",")
            voice_length += float(end) - float(start)
    if os.path.exists(f"{save_folder_path}/output.txt"):
        os.remove(f"{save_folder_path}/output.txt")
    return vad_output_path, voice_length


def get_mean_score(file_emb):
    final_score = 0
    for model_type in cfg.ENCODE_MODEL_LIST:
        emb1 = list(file_emb[model_type]["embedding"].values())[0]
        emb2 = list(file_emb[model_type]["embedding"].values())[1]
        score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        logger.info(f"Model: {model_type}, score: {score}")
        final_score += score
    final_score /= len(cfg.ENCODE_MODEL_LIST)
    logger.info(f"Final score: {final_score}")
    return final_score


@app.route('/compare', methods=['POST'])
def compare():
    try:
        need_vad = request.form.get('need_vad', 1)
        spkid = request.form.get('spkid')
        spkid = spkid.replace('_', '')
        if not spkid:
            return jsonify({"code": 500, "msg": "请指定<spkid>"})
        receive_path = cfg.RECEIVE_PATH
        temp_folder = os.path.join(receive_path, str(spkid))
        os.makedirs(temp_folder, exist_ok=True)

        channel = int(request.form.get('channel', 0))
        window_size = float(request.form.get('window_size'))
        if not window_size:
            return jsonify({"code": 500, "msg": "请指定截取时长<window_size>"})
        uploaded_files = request.files.getlist("wav_files")
        if not uploaded_files or len(uploaded_files) != 2:
            return jsonify({"code": 500, "msg": "请上传需要对比的两个音频文件<wav_files>"})

        local_file_list = []
        raw_file_list = []
        vad_length_list = []
        for filedata in uploaded_files:
            filepath = save_file(filedata, spkid, channel)
            if need_vad == 0:
                local_file_list.append(filepath)
                raw_file_list.append(filepath)
                continue
            raw_sr = torchaudio.load(filepath)[1]
            if raw_sr != 8000:
                random.seed(123)
                _shift = -0.27+0.001*random.random()
            else:
                _shift = 0.0
            try:
                vad_output_path, vad_length = energybase_vad(filepath, temp_folder)
            except Exception as e:
                logger.error(f'vad failed. spkid: {spkid}, Exception: {e}')
                return jsonify({"code": 500, "spkid": spkid, "msg": "音频vad失败。详情:{}".format(e)})
            logger.info(f"VAD output: {vad_output_path}, vad_length: {vad_length}")
            if float(vad_length) < float(window_size):
                return jsonify({"code": 500, "msg": "音频有效时长:{}秒,小于指定的截取时长:{}秒".format(vad_length, window_size)})

            local_file_list.append(vad_output_path)
            raw_file_list.append(vad_output_path)
            vad_length_list.append(vad_length)
        random.seed(123)
        start = random.randint(0, int(min(vad_length_list) - window_size))
        end = start + window_size
        try:
            file_emb = encode_files(local_file_list, raw_file_list, start, end)
        except Exception as e:
            logger.error(f'encode failed. spkid: {spkid}, Exception: {e}')
            return jsonify({"code": 500, "spkid": spkid, "msg": "音频编码失败。详情:{}".format(e)})

        score = get_mean_score(file_emb)
        score += _shift

        return jsonify({
            "code": 200,
            "score": score,
            "msg": "success"})
    except Exception as e:
        logger.error(f'spkid: {spkid}, Exception: {e}')
        return jsonify({"code": 500, "spkid": spkid, "msg": "比对失败。详情:{}".format(e)})
    finally:
        shutil.rmtree(temp_folder)
        torch.cuda.empty_cache()


@app.route("/register", methods=["POST"])
def register():
    """
    register
    """
    try:
        need_vad = request.form.get('need_vad', 1)
        spkid = request.form.get('spkid')
        spkid = spkid.replace('_', '')
        if not spkid:
            return jsonify({"code": 500, "msg": "请指定<spkid>"})
        receive_path = cfg.RECEIVE_PATH
        temp_folder = os.path.join(receive_path, str(spkid))
        os.makedirs(temp_folder, exist_ok=True)

        window_size = float(request.form.get('window_size'))
        if not window_size:
            return jsonify({"code": 500, "msg": "请指定截取时长<window_size>"})

        channel = request.form.get('channel', 0)
        filedata = request.files.get('wav_files')
        if not filedata:
            return jsonify({"code": 500, "msg": "请上传音频文件<wav_files>"})
        t1 = time.time()
        filepath = save_file(filedata, spkid, channel)
        logger.info('*'*20 + f"save file time: {time.time() - t1}")
        if need_vad == 1:
            t1 = time.time()
            raw_sr = torchaudio.load(filepath)[1]
            # if raw_sr > 8000:
            #     spkid = "M@"+spkid
            # else:
            #     spkid = "P@"+spkid
            vad_output_path, vad_length = energybase_vad(filepath, temp_folder)
            logger.info('*'*20 + f"vad time: {time.time() - t1}")

            if vad_length < cfg.VAD_LENGTH:
                return jsonify({"code": 500, "msg": "音频有效时长小于{}秒".format(cfg.VAD_LENGTH)})

            if float(vad_length) < float(window_size):
                return jsonify({"code": 500, "msg": "音频有效时长:{}秒,小于指定的截取时长:{}秒".format(vad_length, window_size)})

            t1 = time.time()
            random.seed(123)
            start = random.randint(0, int(vad_length - window_size))
            end = start + window_size
            try:
                file_emb = encode_files([vad_output_path], [vad_output_path], start, end, need_list=True)
                logger.info('*'*20 + f"encode time: {time.time() - t1}")
            except Exception as e:
                logger.error(f"encode failed. spkid:{spkid}.msg:{e}")
                return jsonify({"code": 500, "spkid": spkid, "msg": "音频编码失败。详情:{}".format(e)})
        else:
            file_length = torchaudio.info(filepath).num_frames / torchaudio.info(filepath).sample_rate
            random.seed(123)
            start = random.randint(0, int(file_length - window_size))
            end = start + window_size
            file_emb = encode_files([filepath], [filepath], start, end, need_list=True)
            vad_length = file_length

        t1 = time.time()
        for model_type in cfg.ENCODE_MODEL_LIST:
            emb_new = list(file_emb[model_type]["embedding"].values())[0]
            to_database(emb_new, spkid=spkid, use_model_type=model_type, mode="register")

        logger.info('*'*20 + f"save to database time: {time.time() - t1}")
        logger.info(f"Register success. spkid:{spkid}")
        return jsonify({"code": 200, "spkid": spkid, "msg": "音频注册成功", "embedding": file_emb, "vad_length": vad_length})

    except Exception as e:
        logger.error(f"Register failed. spkid:{spkid}.msg:{e}")
        return jsonify({"code": 500, "spkid": spkid, "msg": "注册失败。详情:{}".format(e)})
    finally:
        shutil.rmtree(temp_folder)
        torch.cuda.empty_cache()


@app.route('/embeddings', methods=['POST'])
def embeddings():
    need_vad = request.form.get('need_vad', 1)
    spkid = request.form.get('spkid')
    spkid = spkid.replace('_', '')
    if not spkid:
        return jsonify({"code": 500, "msg": "请指定<spkid>"})
    receive_path = cfg.RECEIVE_PATH
    temp_folder = os.path.join(receive_path, str(spkid))
    os.makedirs(temp_folder, exist_ok=True)

    channel = int(request.form.get('channel', 0))
    window_size = float(request.form.get('window_size'))
    if not window_size:
        return jsonify({"code": 500, "msg": "请指定截取时长<window_size>"})

    channel = request.form.get('channel', 0)
    filedata = request.files.get('wav_files')
    if not filedata:
        return jsonify({"code": 500, "msg": "请上传音频文件<wav_files>"})
    t1 = time.time()
    filepath = save_file(filedata, spkid, channel)
    logger.info('*'*20 + f"save file time: {time.time() - t1}")
    if need_vad == 1:
        vad_output_path, vad_length = energybase_vad(filepath, temp_folder)
        logger.info('*'*20 + f"vad time: {time.time() - t1}")

        if vad_length < cfg.VAD_LENGTH:
            return jsonify({"code": 500, "msg": "音频有效时长小于{}秒".format(cfg.VAD_LENGTH)})

        if float(vad_length) < float(window_size):
            return jsonify({"code": 500, "msg": "音频有效时长:{}秒,小于指定的截取时长:{}秒".format(vad_length, window_size)})
        random.seed(123)
        start = random.randint(0, int(vad_length - window_size))
        end = start + window_size
        try:
            file_emb = encode_files([vad_output_path], [vad_output_path], start, end, need_list=True)
        except Exception as e:
            logger.error(f"encode failed. spkid:{spkid}.msg:{e}")
            return jsonify({"code": 500, "spkid": spkid, "msg": "音频编码失败。详情:{}".format(e)})
    else:
        file_length = torchaudio.info(filepath).num_frames / torchaudio.info(filepath).sample_rate
        random.seed(123)
        start = random.randint(0, int(file_length - window_size))
        end = start + window_size
        file_emb = encode_files([filepath], [filepath], start, end, need_list=True)
        vad_length = file_length
    return jsonify({"code": 200, "spkid": spkid, "msg": "音频编码成功", "embedding": file_emb, "vad_length": vad_length})


@app.route("/search", methods=["POST"])
def search():
    """
    search
    """
    try:
        spkid = request.form.get('spkid')
        spkid = spkid.replace('_', '')
        if not spkid:
            return jsonify({"code": 500, "msg": "请指定<spkid>"})
        receive_path = cfg.RECEIVE_PATH
        temp_folder = os.path.join(receive_path, str(spkid))
        os.makedirs(temp_folder, exist_ok=True)

        window_size = float(request.form.get('window_size'))
        if not window_size:
            return jsonify({"code": 500, "msg": "请指定截取时长<window_size>"})

        channel = request.form.get('channel', 0)
        filedata = request.files.get('wav_files')
        if not filedata:
            return jsonify({"code": 500, "msg": "请上传音频文件<wav_files>"})
        filepath = save_file(filedata, spkid, channel)
        raw_sr = torchaudio.load(filepath)[1]
        if raw_sr != 8000:
            random.seed(123)
            _shift = -0.27+0.001*random.random()
        else:
            _shift = 0.0
        t1 = time.time()
        vad_output_path, vad_length = energybase_vad(filepath, temp_folder)
        logger.info('*'*20 + f"vad time: {time.time() - t1}")
        if float(vad_length) < float(window_size):
            return jsonify({"code": 500, "msg": "音频有效时长:{}秒,小于指定的截取时长:{}秒".format(vad_length, window_size)})
        random.seed(123)
        start = random.randint(0, int(vad_length - window_size))
        end = start + window_size
        t1 = time.time()
        file_emb = encode_files([vad_output_path], [vad_output_path], start, end)
        logger.info('*'*20 + f"encode time: {time.time() - t1}")

        register_list = list(get_spkid())

        t1 = time.time()
        final_score_list = []
        for i in register_list:
            embeddings_db = get_embeddings_for_spkid(i)
            final_score = 0
            for model_type in cfg.ENCODE_MODEL_LIST:
                emb_db = embeddings_db[model_type]
                emb_new = list(file_emb[model_type]["embedding"].values())[0]
                score = np.dot(emb_new, emb_db) / (np.linalg.norm(emb_new) * np.linalg.norm(emb_db))
                score = score + _shift
                final_score += score
                logger.info(f"Model: {model_type}, score: {score}")
            final_score /= len(cfg.ENCODE_MODEL_LIST)
            logger.info(f"Final score: {final_score}")
            final_score_list.append([i, final_score])

        final_score_list = sorted(final_score_list, key=lambda x: x[1], reverse=True)
        # get top 10
        final_score_list = final_score_list[:10]
        logger.info(f"top_10: {final_score_list}")
        logger.info('*'*20 + f"search time: {time.time() - t1}")
        return jsonify({"code": 200, "spkid": spkid, "msg": "音频搜索成功", "top_10": final_score_list})

    except Exception as e:
        logger.error(f"Search failed. spkid:{spkid}.msg:{e}")
        return jsonify({"code": 500, "spkid": spkid, "msg": "搜索失败。详情:{}".format(e)})
    finally:
        shutil.rmtree(temp_folder)
        torch.cuda.empty_cache()


# def qwerasdf():
    # app.run(host='0.0.0.0', port=5001)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)