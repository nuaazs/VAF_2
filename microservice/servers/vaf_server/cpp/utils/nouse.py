# -*- coding: utf-8 -*-
import os
def cp_file(file_path,fold_path,savepath):
    """将文件进行重采样后复制到目标目录
    Args:
        file_path (str): 文件完整路径
        fold_path (str): 原始文件夹路径
        savepath (str): 目标文件夹路径
    Returns:
        None
    """
    # save path
    # 保存路径保持与原始文件相同的格式，方便后续处理
    # 比如原始文件为：/datasets_hdd/datasets/cjsd_download/0001/0001_0001_0001_0001.wav ，
    # fold_path为/datasets_hdd/datasets/cjsd_download ，
    # savepath为/datasets_hdd/datasets/cjsd_vad_0.1_0.1
    # 则保存为：/datasets_hdd/datasets/cjsd_vad_0.1_0.1/0001/0001_0001_0001_0001.wav
    rel_path = os.path.relpath(file_path, fold_path)
    print(rel_path)
    save_dir = os.path.join(savepath, os.path.dirname(rel_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wav_path = os.path.join(savepath, rel_path)
    print(wav_path)
if __name__ == "__main__":
    cp_file("/datasets_hdd/datasets/a/0001/0001_0001_0001_0001.wav","/datasets_hdd/datasets/a","/datasets_hdd/datasets/c")