# coding = utf-8
# @Time    : 2022-09-05  15:12:42
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Preprocess.

from utils.preprocess.check_clip import check_clip
from utils.preprocess.save import save_url
from utils.preprocess.save import save_file
from utils.preprocess.vad import vad
from utils.preprocess.resample import resample
from utils.preprocess.classify import classify
from utils.preprocess.mydenoiser import denoise_wav
from utils.preprocess.remove_segments import remove
# from utils.preprocess.remove_fold import remove_fold_and_file
from utils.preprocess.resample import read_wav_data
# from utils.preprocess.mandarin_filter import filter_mandarin