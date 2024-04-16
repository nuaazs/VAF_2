'''
Descripttion: 
version: 1.0.1
Author: duanyibo 444689158@qq.com
Date: 2024-04-01 11:03:49
LastEditors: duanyibo 444689158@qq.com
LastEditTime: 2024-04-02 13:47:57
'''
# 读取一个文件夹下所有.score文件,文件的每行格式为：17002110085_voice_e0d1f2fdf3fb44e5ae279bc058679d08.wav 17002110085_voice_de34d97cd4be46eb9ef8c52ed0dec260_rensheng.wav nontarget 0.26457
# 如果某一行的label = nontarget ，但是得分高于0.5 或者 label = target，但是得分低于0.5，那么将这一行保存到error.txt文件中
import os
import glob
def get_error(file_path):
    error_lines = []
    files = glob.glob(os.path.join(file_path, '*.score'))
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = line.strip().split()
                if line[2] == 'nontarget' and float(line[3]) > 0.5:
                    error_lines.append(line)
                elif line[2] == 'target' and float(line[3]) < 0.5:
                    error_lines.append(line)
    return error_lines

def save_error(error_lines, save_path):
    with open(save_path, 'w') as f:
        for line in error_lines:
            f.write(' '.join(line) + '\n')

if __name__ == '__main__':
    # file_path是一个文件夹，里面有很多.score文件
    file_path = '/VAF/test/result/cti_all_228/resnet101_cjsd_and_resnet221_cjsd_lm_and_resnet293_cjsd_lm/cjsdv2pro/10/scores'
    save_path = './10s_real_fusion_v2pro_error.txt'
    error_lines = get_error(file_path)
    # 对error_lines按照score进行从大到小排序
    error_lines = sorted(error_lines, key=lambda x: float(x[3]), reverse=True)
    save_error(error_lines, save_path)