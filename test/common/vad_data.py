from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import glob
import os
import soundfile
import torchaudio
import torch

# Initialize the VAD pipeline
inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
)

input_dir = '/datasets/test/changjiang_longyuan_test_data/cjsd_change_voice/num_16k_real'
output_dir = '/datasets/test/changjiang_longyuan_test_data/cjsd_change_voice/num_16k_real_denoised'
os.makedirs(output_dir, exist_ok=True)

# Process each WAV file in the directory
for wav_file in glob.glob(os.path.join(input_dir, '*.wav')):
    segments_result = inference_pipeline(audio_in=wav_file)
    wav_data,sr = torchaudio.load(wav_file)
    wav_data = wav_data.reshape(-1)
    print(wav_data.shape)
    output_data = []
    print(segments_result)
    for seg in segments_result["text"]:
        start_second = seg[0]/1000
        end_second = seg[1]/1000
        start_sample = start_second*sr
        end_sample = end_second*sr
        output_data.extend(wav_data[int(start_sample):int(end_sample)])
    # print(output_data)
    print(len(output_data)/16000)
    # save audio
    save_path = wav_file.replace('num_16k_real','num_16k_real_denoised')
    output_data = torch.tensor(output_data).reshape(1,-1)
    torchaudio.save(save_path,output_data,sr)
print("Processing complete.")
