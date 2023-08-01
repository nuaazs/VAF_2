# 版本要求 modelscope version >= 1.7.0
from modelscope.pipelines import pipeline
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)
input_wav = '/home/zhaosheng/2p_16k.wav'
result = sd_pipeline(input_wav)
print(result)
# 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
result = sd_pipeline(input_wav, oracle_num=2)
print(result)
# 如果发现cpu推理过慢的情况，可调小batch_size参数
result = sd_pipeline(input_wav, batch_size=1)
print(result)