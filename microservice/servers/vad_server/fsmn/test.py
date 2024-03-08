
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

import torch


inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
    device='cuda:0'
)

segments_result = inference_pipeline(audio_in='/VAF/microservice/servers/vad_server/fsmn/raw_13913009681.wav')
print(segments_result)