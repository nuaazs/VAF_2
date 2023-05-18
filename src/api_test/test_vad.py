# @Time    : 2022-07-27  18:58:05
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : http://www.iint.icu/
# @File    : /mnt/zhaosheng/VAF-System/src/utils/preprocess.py
# @Describe: Preprocess wav files.

import torch
import torchaudio.transforms as T
import os
import soundfile as sf
import time
import torchaudio
import cfg

USE_ONNX = True

model, utils = torch.hub.load(repo_or_dir='./snakers4_silero-vad_master',
                             source='local',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# 打印函数时间装饰器
def print_time(func):
    def inner(*args, **kwargs):
        old_time = time.time()
        result = func(*args, **kwargs)
        func_name = str(func).split(' ')[1]
        print('{} use time: {}s'.format(func_name, time.time() - old_time))
        return result

    return inner

@print_time
def vad_and_upsample(wav_file,spkid,wav_length=90,savepath=None,channel=1,save_days=30):
    """vad and upsample to 16k.

    Args:
        wav_file (string): filepath of the uploaded wav file.
        channel (int, optional): which channel to use. Defaults to 0.

    Returns:
        torch.tensor: new wav tensor
    """
    local_time = time.time()
    wav, sr = torchaudio.load(wav_file)
    print(wav)
    print(wav.shape)


    wav = torch.FloatTensor(wav[channel,:])
   
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        wav = resampler(wav)
    print(wav)
    print(wav.shape)
    before_vad_length = len(wav)/sr
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000,window_size_samples=1536)
    if len(speech_timestamps)>1:
        wait_time = (speech_timestamps[1]["start"] - speech_timestamps[0]["end"])/16000
    else:
        wait_time = 0
    wav_torch = collect_chunks(speech_timestamps, wav)
    if not savepath:
        savepath = cfg.TEMP_PATH
    spk_dir = os.path.join(savepath, str(spkid))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)
    speech_number = len(spk_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm
    save_name = f"preprocessed_{spkid}_{speech_number}.wav"
    final_save_path = os.path.join(spk_dir, save_name)
    save_audio(final_save_path,wav_torch, sampling_rate=16000)
    
    after_vad_length = len(wav_torch)/16000.
    used_time = time.time() - local_time
    result = {
        "wav_torch":wav_torch,
        "before_length":before_vad_length,
        "after_length":after_vad_length,
        "save_path":final_save_path,
        "used_time":used_time,
        "wait_time":wait_time,

    }
    return result

@print_time
def self_test(wav_torch, spkreg,similarity, sr=16000, min_length=5, similarity_limit=0.60):
    """Quality detection function, self-splitting into multiple fragments and then testing them in pairs.

    Args:
        wav_torch (torch.tensor): input wav
        spkreg (speechbarin.model): embedding model from speechbrain.
        similarity (function): similarity function
        sr (int, optional): sample rate. Defaults to 16000.
        split_num (int, optional): split wav to <num> fragments. Defaults to 3.
        min_length (int, optional): length(s) of each fragment. Defaults to 3.
        similarity_limit (float, optional): similarity limit for self-test. Defaults to 0.7.

    Returns:
        _type_: pass or not, message
    """
    local_time = time.time()
    max_score = 0
    min_score = 1

    if len(wav_torch)/sr <= min_length:
        used_time = time.time() - local_time
        result = {
            "pass":False,
            "msg":f"Insufficient duration, the current duration is {len(wav_torch)/sr}s.",
            "max_score":0,
            "mean_score":0,
            "min_score":0,
            "used_time":used_time,
        }
        return result

    half_length = int(len(wav_torch)/2)
    tiny_wav1 = torch.tensor(wav_torch[half_length:]).unsqueeze(0)
    embedding1 = spkreg.encode_batch(tiny_wav1)[0][0]

    tiny_wav2 = torch.tensor(wav_torch[:half_length]).unsqueeze(0)
    embedding2 = spkreg.encode_batch(tiny_wav2)[0][0]
    score = similarity(embedding1, embedding2).numpy()
    max_score,mean_score,min_score = score,score,score
    used_time = time.time() - local_time

    if score < similarity_limit:
        result = {
            "pass":False,
            "msg":f"Bad quality score:{min_score}.",
            "max_score":max_score,
            "mean_score":mean_score,
            "min_score":min_score,
            "used_time":used_time,
        }
        return result
    result = {
            "pass":True,
            "msg":"Qualified.",
            "max_score":max_score,
            "mean_score":mean_score,
            "min_score":min_score,
            "used_time":used_time,
        }
    return result


if __name__ == "__main__":
    vad_and_upsample(wav_file="/mnt/cti_record_data_with_phone_num/13321580582/raw_1.wav",spkid="123",wav_length=90,savepath=None,channel=1,save_days=30)