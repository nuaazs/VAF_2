# Function: split 6s audio from 1min audio



import os
import wave

def split_audio(input_path, output_path, txt_path):
    # 读取文本文件
    with open(txt_path, 'r') as txt_file:
        splits_raw = txt_file.readlines()

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)


    # 计算每段音频的帧数（6秒）
    # frames_per_segment = int(6 * frame_rate)
    len_limit = 6
    i = 0
    durations=[]
    print(f"len(splits_raw): {len(splits_raw)}")
    print(f"splits_raw: {splits_raw}")
    durations_result = []
    split_result = []

    for split_raw in splits_raw:
        durations.append(float(split_raw.split(",")[1])-float(split_raw.split(",")[0]))
    print(f"durations: {durations}")
    print(f"len(durations): {len(durations)}")
    start_time = 0
    while i < len(durations):
        
        end_time = float(start_time+durations[i])

        current_duration = (end_time - start_time)
        while current_duration < len_limit:
            if i < len(durations)-1:
                i += 1
                print(f"i: {i}")
                end_time = float(end_time+durations[i])
                current_duration = (end_time - start_time)
            else:
                break
            
            
        durations_result.append((start_time, end_time))
        split_result.append(current_duration)
        # use ffmpeg get input_path audio segment
        fid = input_path.split('/')[-1].split('.')[0]
        os.makedirs(os.path.join(output_path, fid), exist_ok=True)
        start_time = end_time
        i += 1
        print(f"Now i: {i}")
        print(f"durations_result: {durations_result}")

    count = 0
    for split in durations_result:
        start_time = split[0]
        end_time = split[1]
        print("="*20)
        print(f"start_time: {start_time}, end_time: {end_time}")
        cmd = f"ffmpeg -ss {start_time:.1f} -to {end_time:.1f} -i {input_path} -acodec copy {output_path}/{fid}/{fid}_{count}.wav -y > /dev/null 2>&1"#  
        print(cmd)
        os.system(cmd)
        start_time = end_time
        count += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vad_dir', type=str,help='')
    parser.add_argument('--split_dir', type=str,help='')
    parser.add_argument('--info_dir', type=str,help='')
    args = parser.parse_args()

    # 遍历data_vad目录下的所有音频文件和对应的txt文件
    data_vad_dir = args.vad_dir # "data_vad"
    data_vad_split_dir = args.split_dir #"data_vad_split"
    info_dir = args.info_dir #"data_vad_split"

    for phone_number in os.listdir(data_vad_dir):
        phone_dir = os.path.join(data_vad_dir, phone_number)
        for file_name in os.listdir(phone_dir):
            if file_name.endswith(".wav"):
                wav_path = os.path.join(phone_dir, file_name)
                os.makedirs(os.path.join(info_dir, phone_number), exist_ok=True)
                txt_path = os.path.join(info_dir, phone_number, os.path.splitext(file_name)[0] + ".txt")
                output_dir = os.path.join(data_vad_split_dir, phone_number)
                os.makedirs(output_dir, exist_ok=True)
                split_audio(wav_path, output_dir, txt_path)

# split_audio("/home/zhaosheng/Documents/cjsd_train_data/data_vad/13195751289/20210913112545.wav", "data_vad_split/13195751289", "/home/zhaosheng/Documents/cjsd_train_data/data_info/13195751289/20210913112545.txt")
