from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')

result = ans(
    '/datasets/icassp/noise_audio1.wav',
    output_path='denoise_audio1.wav')
result = ans(
    '/datasets/icassp/noise_audio2.wav',
    output_path='denoise_audio2.wav')
result = ans(
    '/datasets/icassp/noise_audio3.wav',
    output_path='denoise_audio3.wav')
result = ans(
    '/datasets/icassp/noise_audio4.wav',
    output_path='denoise_audio4.wav')