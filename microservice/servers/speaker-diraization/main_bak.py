
from flask import Flask, request, jsonify
import os

import torch
import cfg
from utils.preprocess.save import save_file, save_url
from utils.vad import nn_vad, energybase_vad
from utils.encoder import encode_folder
from utils.cluster import find_optimal_subset
from utils.mandarin import mandarin_filter
from utils.oss.upload import upload_file, upload_files, remove_urls_from_bucket
from asr import file_upload_offline as asr_offline
from utils.nlp import classify_text
import subprocess
import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)


def pipeline(filepath, vad_type, spkid):
    # return example
    tmp_folder = f"/tmp/{spkid}"
    os.makedirs(tmp_folder, exist_ok=True)
    # 步骤一： VAD
    if vad_type == 'nn_vad':
        wav_files = nn_vad(filepath, tmp_folder)
    elif vad_type == 'energybase_vad':
        wav_files = energybase_vad(filepath, tmp_folder)
    else:
        return None
    vad_urls = upload_files(bucket_name="testing",
                            files=wav_files, save_days=180, folder_name=spkid)
    vad_result = [f"文件: {wav}" for wav in wav_files]
    vad_audios = wav_files

    print(f"\t * -> VAD结果: {vad_result}")

    # 步骤二： 普通话过滤
    wav_files=wav_files[0:2]
    mandarin_wavs = mandarin_filter(wav_files)
    mandarin_urls = upload_files(bucket_name="testing",
                                 files=mandarin_wavs, save_days=180, folder_name=spkid)
    mandarin_result = [f"文件: {wav}" for wav in mandarin_wavs]
    mandarin_audios = mandarin_wavs
    print(f"\t * -> 普通话过滤结果: {mandarin_result}")

    # 步骤三： 提取特征
    file_emb = encode_folder(mandarin_wavs)
    print(f"\t * -> 提取特征结果: {file_emb}")

    # 步骤四： 聚类
    selected_files, total_duration, url, wav_file_path, selected_times = find_optimal_subset(
        file_emb, spkid=spkid, similarity_threshold=0.8, save_wav_path=tmp_folder)
    print(f"\t * -> 聚类结果: {selected_times}")
    print(f"\t * -> 聚类结果URL: {url}")
    selected_urls = upload_files(bucket_name="testing",
                                 files=selected_files, save_days=180, folder_name=spkid)

    # 步骤五： ASR
    text = asr_offline(filepath)
    print(f"\t * -> ASR结果: {text}")

    # 步骤六： NLP
    nlp_result = classify_text(text)
    print(f"\t * -> 文本分类结果: {nlp_result}")

    subprocess.call(f"rm -rf {tmp_folder}", shell=True)
    return {
        "raw_file_path": filepath,
        "vad_result": vad_result,
        "vad_urls": vad_urls,
        "play_vad_files": vad_audios,
        "mandarin_filter_result": mandarin_result,
        "mandarin_urls": mandarin_urls,
        "play_mandarin_files": mandarin_audios,
        "embeddings": file_emb,
        "selected_times": selected_times,
        "url": url,
        "asr_result": text,
        "nlp_result": nlp_result,
        "total_duration": total_duration,
        "selected_urls": selected_urls,


    }


def process_audio(input_file, vad_type, spkid):
    output = {}
    output["file_name"] = input_file
    output["audio_length"] = "待计算"
    # output["play_audio"] = gr.outputs.Audio("filepath")

    pipeline_result = pipeline(input_file, vad_type, spkid)
    # save_png_path = f"./png/{pipeline_result['raw_file_path'].split('/')[-1].replace('.wav','.png')}"
    # os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    # get_spectrogram(pipeline_result['raw_file_path'], save_png_path)
    output["audio_length"] = pipeline_result["total_duration"]
    output["play_audio"] = pipeline_result["raw_file_path"]

    vad_urls = pipeline_result.get("vad_urls", [])
    vad_urls = sorted(vad_urls, key=lambda x: float(
        x.split("/")[-1].split("_")[0]))
    output["vad_urls"] = vad_urls
    output["mandarin_urls"] = pipeline_result.get("mandarin_urls", [])
    output["url"] = pipeline_result.get("url", '')
    output["asr_result"] = pipeline_result.get("asr_result", '')['text']
    output["nlp_result"] = pipeline_result.get("nlp_result", '')
    # vad_players = create_audio_players(vad_urls)
    # output["vad_urls"] = generate_html_content(vad_players, "VAD结果")

    # output["play_vad_files"] = pipeline_result.get("play_vad_files", [])

    selected_urls = pipeline_result.get("selected_urls", [])
    selected_urls = sorted(selected_urls, key=lambda x: float(
        x.split("/")[-1].split("_")[0]))
    output["selected_urls"] = selected_urls
    # selected_players = create_audio_players(selected_urls)
    # output["selected_result"] = generate_html_content(selected_players, "选择的片段")

    return output


@app.route('/speaker-diraization/<filetype>', methods=['POST'])
def main(filetype):
    try:
        spkid = request.form.get('spkid')
        channel = int(request.form.get('channel', 0))
        save_oss = request.form.get('save_oss', False)
        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        tmp_folder = f"/tmp/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)

        if filetype == "file":
            filedata = request.files.get('file')
            filepath, url = save_file(
                filedata, spkid, channel, upload=save_oss)
            print(filepath)
        elif filetype == "url":
            url = request.form.get('url')
            filepath, url = save_url(url, spkid, channel, upload=save_oss)
        return_data = process_audio(filepath, "nn_vad", spkid)
        return_data['code'] = 200
        return_data['file_url'] = url
        return jsonify(return_data)
    except Exception as e:
        print(e)
        return jsonify({"code": 500, "message": str(e)})
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7001)
