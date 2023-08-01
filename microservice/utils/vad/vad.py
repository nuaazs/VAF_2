import subprocess
import os

def energybase_vad(filepath,save_folder_path,smooth_threshold=0.5,min_duration=2):
    os.makedirs(save_folder_path,exist_ok=True)
    bin_path = f"{filepath.split('/')[-1][:-4]}.bin"
    bin_path = os.path.join(save_folder_path,bin_path)
    cmd = f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  {bin_path}> /dev/null 2>&1"
    subprocess.call(cmd,shell=True)
    vad_cmd = f"vad {bin_path} {save_folder_path} {smooth_threshold} {min_duration}"
    subprocess.call(vad_cmd,shell=True)
    file_list = [os.path.join(save_folder_path,_file) for _file in os.listdir(save_folder_path) if _file.endswith(".wav")]
    return file_list

if __name__ == "__main__":
    energybase_vad("/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav","./temp/")