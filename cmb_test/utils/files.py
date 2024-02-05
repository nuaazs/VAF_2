# coding = utf-8
# @Time    : 2023-10-24  13:19:10
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: wav files utils.
import os
import glob
def find_wav_files(root,pattern='.wav'):
    '''
    Find all wav files in root.
    Args:
        root: root path
        pattern: partten of wav files
    Returns:
        wav_files: wav files list
    '''
    wav_files = []
    # find all wav files use glob
    wavs = glob.glob(os.path.join(root,'**',pattern),recursive=True)
    # filter wav files
    for wav in wavs:
        if os.path.isfile(wav):
            wav_files.append(wav)
    return wav_files