# -*- coding: utf-8 -*-
'''
@author: Carry
@contact: xkx94317@gmail.com
@file: make_excel.py
@time: 2019/5/17 18:11 
@desc: 汇总 制作 excel表格
'''
import os
import shutil
import statistics

method_li = ['campp_0103_all_data',
             'campp_0103_female_data',
             'campp_0103_male_data',
             'ecapatdnn_0101_all_data',
             'ecapatdnn_0101_female_data',
             'ecapatdnn_0101_male_data',
             'ecapatdnn_0103_all_data',
             'ecapatdnn_0103_female_data',
             'ecapatdnn_0103_male_data']


# p = "ecapatdnn_campp_mean_all_data.score"
# p = "ecapatdnn_campp_mean_female_data.score"
# p = "ecapatdnn_campp_mean_male_data.score"


def main(total):
    black_li = []
    test_li = []
    a_set = set()
    std = []
    same = diff = count = 0
    base_path = f"/home/xz/zhaosheng/get_cjsd_embeddings/{p}/input_a/vector_a_all_split_data"

    # with open(f'/home/xz/zhaosheng/get_cjsd_embeddings/{p}', 'r') as f:
    with open(base_path + '/all.score', 'r') as f:
        for idx, i in enumerate(f.readlines()):
            if idx == total:
                break
            phone_a = i.split(',')[0].split('_')[0]
            phone_b = i.split(',')[1].split('_')[0]
            score = i.split(',')[-1]
            count += float(score)
            std.append(float(score))
            if 'embedd' not in i.split(',')[0]:
                same += 1
            else:
                diff += 1
                black_li.append(i.split(',')[0] + '.wav')
                test_li.append(i.split(',')[1] + '.wav')
                # print(i.strip())
                a_set.add(f"{phone_a}_{phone_b}")
    print(f"acc:{same / total},mean:{count / total},std:{round(statistics.stdev(std), 4)}")
    acc = round(same / total, 2)
    mean_score = round(count / total, 2)
    std = round(statistics.stdev(std), 4)
    ss = f"{acc},{mean_score},{std},"
    print(ss)
    return ss


if __name__ == '__main__':
    if os.path.exists('output.csv'):
        os.remove('output.csv')
    for idx, p in enumerate(method_li):
        output_str = f"{p.split('_')[0]},{p.split('_')[1]},{p.split('_')[2]},"
        for top in [10, 20, 30, 50, 100, 200]:
            output_str += main(top)
        with open(f'output.csv', 'a') as f:
            if idx == 0:
                f.write(
                    '模型名称,参数,性别,top10_acc,top10_mean_score,top10_std,top20_acc,top20_mean_score,top20_std,'
                    'top30_acc,top30_mean_score,top30_std,top50_acc,top50_mean_score,top50_std,'
                    'top100_acc,top100_mean_score,top100_std,top200_acc,top200_mean_score,top200_std' + '\n')
            f.write(output_str + '\n')
