from flask import Flask, request, jsonify
import os
import logging
import torch
import soundfile
import torchaudio

from utils.files import get_sub_wav
from utils.cmd import run_cmd
from utils.preprocess import save_file, save_url
from utils.oss import upload_files
from utils.log import logger, err_logger
import cfg
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# logger = get_logger(log_level=logging.CRITICAL)
# logger.setLevel(logging.CRITICAL)
# pwd = os.path.dirname(os.path.abspath(__file__))
# print(f'models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',)

# offline_inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model=f'models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
#     model_revision="v1.2.4",
#     device=cfg.ASR_PYTHON_DEVICE,
# )

param_dict = dict()
param_dict['hotword'] = "./hotword.txt"
param_dict['use_timestamp'] = False
offline_inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
             #damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    # vad_model='models/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    # vad_model_revision="v1.1.8",
    # punc_model='models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    param_dict=param_dict,
    device=cfg.ASR_PYTHON_DEVICE,
)

# online_inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model=f'models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
#     model_revision='v1.0.6',
#     mode="paraformer_streaming",
#     device='gpu:3',
# )

inference_pipline = pipeline(
    task=Tasks.punctuation,
    model=f'models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision="v1.1.7",
    device=cfg.PUNC_PYTHON_DEVICE,
    audio_fs=16000)
    
)

# def transcribe_audio(audio):
#     audio = audio.reshape(-1).numpy()
#     sample_offset = 0
#     chunk_size = [5, 10, 5]  # [5, 10, 5] 600ms, [8, 8, 4] 480ms
#     stride_size = chunk_size[1] * 960
#     param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
#     final_result = ""

#     for sample_offset in range(0, len(audio), min(stride_size, len(audio) - sample_offset)):
#         if sample_offset + stride_size >= len(audio) - 1:
#             stride_size = len(audio) - sample_offset
#             param_dict["is_final"] = True
#         rec_result = online_inference_pipeline(audio_in=audio[sample_offset: sample_offset + stride_size],
#                                                param_dict=param_dict)
#         if len(rec_result) != 0:
#             final_result += rec_result['text'] + " "
#             print(rec_result)
#     return final_result


def transcribe_audio_offline(audio):
    audio = audio.reshape(-1)  # .to(cfg.ASR_PYTHON_DEVICE)
    rec_result = offline_inference_pipeline(audio_in=audio)
    return rec_result['text']


app = Flask(__name__)


@app.route("/transcribe/<filetype>", methods=["POST"])
def main(filetype):
    try:
        channel = int(request.form.get('channel', 0))
        save_oss = request.form.get('save_oss', False)
        if save_oss != False and save_oss.lower() in ['true', 'yes', '1']:
            save_oss = True
        postprocess = int(request.form.get('postprocess', 0))
        spkid = request.form.get('spkid', "init_id")

        if filetype == "file":
            filedata = request.files.get('wav_file')
            filepath, url = save_file(filedata, spkid, channel, upload=save_oss)
        else:
            filepath, url = save_url(request.form.get(
                'url'), spkid, channel, upload=save_oss)
        audio, sample_rate = torchaudio.load(filepath)
        logger.info(f"sample_rate: {sample_rate}")
        logger.info(f"Now transcribing {filepath}")
        # try:
        transcription = transcribe_audio_offline(audio)
        # except Exception as e:
        #     err_logger.error(f"Error: {e}")
        #     transcription = ""
        logger.info(f"Transcription: {transcription}")
        if postprocess:
            transcription = inference_pipline(text_in=transcription)
        # empty the cuda cache
        torch.cuda.empty_cache()
        return jsonify({"transcription": transcription})
    except Exception as e:
        err_logger.error(f"Error: {e}")
        torch.cuda.empty_cache()
        return jsonify({"code": 500, "message": str(e)})
    # finally:
    torch.cuda.empty_cache()


@app.route("/")
def home():
    return "Welcome to the transcription service!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
