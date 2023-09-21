# 读取model_a_score_path
# 其内容为：
# phone,blackbase_phone,hit_score
# MTk5NDE4NDYwNzEK,156289624,0.9999999999999999
# MTUzNzA1NDM3MTkK,158150446,0.9999999999999999

# 读取model_b_score_path
# 其内容为：
# phone,blackbase_phone,hit_score
# MTk5NDE4NDYwNzEK,156289624,0.9999999999999999
# MTUzNzA1NDM3MTkK,158150446,0.9999999999999999

# 目的：
# 1. 读取model_a_score_path和model_b_score_path获得两个结果集
# 2. 分别对两个结果集进行正则化
# 3. 对两个正则化后的结果集进行取均值操作
# 4. 对均值结果集进行排序生成合并后结果集保存到output_path
# 5. 注意：只取两个结果集中的重复项，生成重复项结果集保存到output_path，对于在一个结果中有得分，另一个结果中没有得分的项，不保留

import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_a_score_path', type=str, default='ecapatdnn_0103_female_data/input_a/vector_a_all_split_data/all.score',help='')
parser.add_argument('--model_b_score_path', type=str, default='campp_0103_female_data/input_a/vector_a_all_split_data/all.score',help='')
parser.add_argument('--output_path', type=str, default='ecapatdnn_campp_mean_female_data.score',help='')
args = parser.parse_args()

# 读取 model_a_score_path
with open(args.model_a_score_path, 'r') as f:
    a_scores = f.readlines()

# 读取 model_b_score_path
with open(args.model_b_score_path, 'r') as f:
    b_scores = f.readlines()

# 将 a_scores 和 b_scores 中的字符串转化为元组列表
def parse_scores(scores):
    res = []
    for line in scores:
        phone, blackbase_phone, hit_score = line.strip().split(',')
        res.append((phone, blackbase_phone, float(hit_score)))
    return res

a_scores = parse_scores(a_scores)
b_scores = parse_scores(b_scores)

# 将 a_scores 和 b_scores 中的元组按照 (phone, blackbase_phone) 分类，得到字典
def group_scores(scores):
    res = {}
    for score in scores:
        phone, blackbase_phone, hit_score = score
        key = (phone, blackbase_phone)
        # if key in res:
        #     res[key].append(hit_score)
        # else:
        res[key] = [hit_score]
    return res

def get_mean_score(res):
    scores = []
    for _key in res.keys():
        scores.append(res[_key][0])
    return np.mean(scores)

a_grouped_scores = group_scores(a_scores)
a_mean_score = get_mean_score(a_grouped_scores)
b_grouped_scores = group_scores(b_scores)
b_mean_score = get_mean_score(b_grouped_scores)
print(b_mean_score)
# print(a_grouped_scores)
# 对 a_grouped_scores 和 b_grouped_scores 中每个键值对中的值进行正则化，然后再进行平均，得到字典
def normalize_and_average_scores(grouped_scores,mean_value):
    res = {}
    for key, value in grouped_scores.items():
        # print(value)
        normalized_value = [(score - mean_value)  for score in value] # / (max(value) - min(value) + 0.000001)
        res[key] = sum(normalized_value) / len(normalized_value)
    return res

a_norm_avg_scores = normalize_and_average_scores(a_grouped_scores,a_mean_score)
b_norm_avg_scores = normalize_and_average_scores(b_grouped_scores,b_mean_score)

# 获取 a_norm_avg_scores 和 b_norm_avg_scores 中的重复项，生成新字典
intersect_scores = {}
for key in set(a_norm_avg_scores.keys()) & set(b_norm_avg_scores.keys()):
    intersect_scores[key] = (a_norm_avg_scores[key], b_norm_avg_scores[key], (a_norm_avg_scores[key] + b_norm_avg_scores[key])/2)

# 将 intersect_scores 中的元组按照第3个元素（即 (model_b+model_a)/2 的正则化均值得分）的大小排序
sorted_inter_scores = sorted(intersect_scores.items(), key=lambda x: x[1][2], reverse=True)

# 将结果写入 output_path
with open(args.output_path, 'w') as f:
    # 写入表头
    # f.write('phone,blackbase_phone,mean_score\n') # model_a_score,model_b_score,
    # 写入数据
    for item in sorted_inter_scores:
        phone, blackbase_phone = item[0]
        a_score = item[1][0]
        b_score = item[1][1]
        m_score = item[1][2]
        f.write(f'{phone},{blackbase_phone},{m_score}\n')
        # print(f'{phone},{blackbase_phone},{m_score}\n')
