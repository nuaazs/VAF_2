from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import sys
sys.path.append("/home/zhaosheng/asr_damo_websocket/online/speaker-diraization")
import cfg
import subprocess

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model=f'{cfg.damo}/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)

def get_sub_wav(wav_file_path,save_folder,timelist,smooth_threshold,min_duration):
    print(timelist)
    total_number = 0
    # smooth_threshold
    # merge items if the gap between two items is less than smooth_threshold
    for i in range(len(timelist)-1):
        if timelist[i+1][0] - timelist[i][1] < smooth_threshold:
            timelist[i+1][0] = timelist[i][0]
            timelist[i][1] = timelist[i+1][1]
    timelist = list(set([tuple(t) for t in timelist]))
    # remove items if the duration of item is less than min_duration
    timelist = [_item for _item in timelist if _item[1] - _item[0] >= min_duration]
    for item in timelist:
        start = item[0]/1000
        stop = item[1]/1000
        if stop - start < min_duration:
            continue
        filename = f"{start:.2f}_{stop:.2f}.wav"
        save_path = os.path.join(save_folder,filename)
        cmd = f"ffmpeg -i {wav_file_path} -ss {start} -to {stop} -y {save_path} > /dev/null 2>&1"
        subprocess.call(cmd,shell=True)
        total_number = total_number + 1
    file_list = [os.path.join(save_folder,_file) for _file in os.listdir(save_folder) if _file.endswith(".wav") and "_" in _file]
    try:
        file_list.remove(wav_file_path)
        file_list.remove(wav_file_path.replace("_vad.wav",".wav"))
    except:
        pass
    return file_list

        

def nn_vad(filepath,save_folder_path,smooth_threshold=0.5,min_duration=1):
    os.makedirs(save_folder_path,exist_ok=True)
    subprocess.call(f"ffmpeg -i {filepath} -ar 16000 -y  {filepath.replace('.wav','_vad.wav')}> /dev/null 2>&1",shell=True)
    filepath = filepath.replace('.wav','_vad.wav')
    # print(filepath)
    timelist = inference_pipeline(audio_in=filepath,audio_fs=16000).get("text")
    file_list = get_sub_wav(filepath,save_folder_path,timelist,smooth_threshold,min_duration)
    subprocess.call(f"rm -rf {filepath}",shell=True)
    return file_list


if __name__ == "__main__":
    nn_vad("/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav","./temp")