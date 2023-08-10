import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from speakerlabduanyibo.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)
from sklearn.metrics import precision_recall_curve

folder_path = "result_8_7"
paths = glob.glob(f'{folder_path}/**/*.score', recursive=True)

def compute_fnr_fpr(scores, threshold):
    target_scores = scores['target']
    nontarget_scores = scores['nontarget']
    
    num_target = len(target_scores)
    num_nontarget = len(nontarget_scores)
    
    num_miss = sum(score < threshold for score in target_scores)
    num_fa = sum(score >= threshold for score in nontarget_scores)
    
    fnr = num_miss / num_target
    fpr = num_fa / num_nontarget
    
    return fnr, fpr


def get_det_data(path):

    # 省略读取分数和标签的代码
    lines = open(path).readlines()
    scores = {'target': [], 'nontarget': []}
    scores = []
    labels = []
    for line in lines:
        cols = line.split(' ')
        label = cols[2]
        if label == '1' or label == 'target':
            labels.append(1)
        elif label == '0' or label == 'nontarget':
            labels.append(0)
        else:
            raise Exception(f'Unrecognized label in {line}.')
        score = float(cols[3])
        scores.append(score)
    scores = np.array(scores)
    labels = np.array(labels)
    fnrs, fprs = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnrs, fprs, scores)
    # 计算 precision 和 recall
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # 寻找精度首次达到99%的阈值索引
    precision_99_idx = np.argmax(precisions >= 0.99)

    # 获取对应阈值 
    threshold_99 = thresholds[precision_99_idx]
    
    # 根据阈值计算召回率
    pred_99 = (scores >= threshold_99).astype(int)
    recall_99 = np.sum(pred_99[labels==1]) / np.sum(labels)

    print(f'Precision 99% => Recall {recall_99:.2%}') 

    return fprs, fnrs, eer

# def get_det_data(path):
#     lines = open(path).readlines()
#     # scores = {'target': [], 'nontarget': []}
#     scores = []
#     labels = []
#     for line in lines:
#         cols = line.split(' ')
#         label = cols[2]
#         if label == '1' or label == 'target':
#             labels.append(1)
#         elif label == '0' or label == 'nontarget':
#             labels.append(0)
#         else:
#             raise Exception(f'Unrecognized label in {line}.')
#         score = float(cols[3])
#         scores.append(score)
#     scores = np.array(scores)
#     labels = np.array(labels)
#     fnrs, fprs = compute_pmiss_pfa_rbst(scores, labels)
#     eer, thres = compute_eer(fnrs, fprs, scores)
#     min_dcf_noc = compute_c_norm(fnrs,
#                                     fprs,
#                                     p_target=0.0005,
#                                     c_miss=1,
#                                     c_fa=5)
#     print(min_dcf_noc)
    #     scores[label].append(score)
    # target_scores = scores['target'] 
    # nontarget_scores = scores['nontarget']

    # min_score = min(min(target_scores), min(nontarget_scores))
    # max_score = max(max(target_scores), max(nontarget_scores))

    # thresholds = np.arange(min_score, max_score, 0.001)
    # # thresholds = np.arange(min(scores), max(scores), 0.001)  
    # fnrs, fprs = [], []
    
    # for thresh in thresholds:
    #     fnr, fpr = compute_fnr_fpr(scores, thresh)
    #     fnrs.append(fnr)
    #     fprs.append(fpr)
        
    return fprs, fnrs,eer
        
det_data = {}
for path in tqdm(paths):
    model = path.split("/")[-4]
    print(model)
    fprs, fnrs,eer = get_det_data(path)
    # print(eer)
    det_data[model] = (fprs, fnrs)
    
plt.figure()
for model, (fprs, fnrs) in det_data.items():
    plt.plot(fprs, fnrs, label=model)
    # plt.plot(fnrs,fprs,label=model)
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(0, 0.3)
plt.ylim(0, 0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.legend()
plt.title('DET Curves')
plt.savefig('det.png')