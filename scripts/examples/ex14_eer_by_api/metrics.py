# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

def compute_eer(scores: Union[np.ndarray, List[float]],
                labels: Union[np.ndarray, List[int]],
                det_pic_save_path=None,
                roc_pic_save_path=None) -> Tuple[
                    float, float, np.ndarray, np.ndarray]:
    """Compute equal error rate(EER) given matching scores and corresponding labels

    Parameters:
        scores(np.ndarray,list): the cosine similarity between two speaker embeddings.
        labels(np.ndarray,list): the labels of the speaker pairs, with value 1 indicates same speaker and 0 otherwise.

    Returns:
        eer(float):  the equal error rate.
        thresh_for_eer(float): the thresh value at which false acceptance rate equals to false rejection rate.
        fr_rate(np.ndarray): the false rejection rate as a function of increasing thresholds.
        fa_rate(np.ndarray): the false acceptance rate as a function of increasing thresholds.

    scores = [0.13,0.245,0.98 ..... ]
    labels = [1,0,1,1,0, ..... ]
    """
    
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)
    label_set = list(np.unique(labels))
    assert len(
        label_set
    ) == 2, f'the input labels must contains both two labels, but recieved set(labels) = {label_set}'
    label_set.sort()
    assert label_set == [
        0, 1
    ], 'the input labels must contain 0 and 1 for distinct and identical id. '
    eps = 1e-8
    #assert np.min(scores) >= -1.0 - eps and np.max(
    #    scores
    #  ) < 1.0 + eps, 'the score must be in the range between -1.0 and 1.0'
    same_id_scores = scores[labels == 1]
    diff_id_scores = scores[labels == 0]
    thresh = np.linspace(np.min(diff_id_scores), np.max(same_id_scores), 1000)
    thresh = np.expand_dims(thresh, 1)
    fr_matrix = same_id_scores < thresh
    fa_matrix = diff_id_scores >= thresh
    fr_rate = np.mean(fr_matrix, 1)
    fa_rate = np.mean(fa_matrix, 1)
    tr_rate = 1 - fr_rate
    # plot det curve
    thresh_idx = np.argmin(np.abs(fa_rate - fr_rate))
    result = namedtuple('speaker', ('eer', 'thresh', 'fa', 'fr'))
    result.eer = (fr_rate[thresh_idx] + fa_rate[thresh_idx]) / 2
    result.thresh = thresh[thresh_idx, 0]
    result.fr = fr_rate
    result.fa = fa_rate
    
    if det_pic_save_path:
        plt.figure()
        plt.plot(fa_rate, fr_rate)
        plt.xlabel(r'$FA\ Rate$')
        plt.ylabel(r'$FR\ Rate$')

        plt.xscale('function', functions=(forward,inverse))
        plt.yscale('function', functions=(forward,inverse))
        # set x trick
        plt.xticks([0, 0.01, 0.05, 0.1, 0.2, 0.5])
        plt.yticks([0, 0.01, 0.05, 0.1, 0.2, 0.5])
        # set range 0~0.5
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        
        # plot x==y dot line, red
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        # plot eer point, blue with value text
        plt.plot(result.eer, result.eer, 'bo')
        plt.text(result.eer, result.eer, f'(EER={result.eer*100:.2f}%)')
        # plt.yscale('log')
        plt.savefig(det_pic_save_path)
        # plt.show()
    if roc_pic_save_path:
        plt.figure()
        plt.plot(fa_rate, tr_rate)
        plt.xlabel('fa_rate')
        plt.ylabel('tr_rate')
        plt.savefig(roc_pic_save_path)
        # plt.show()
    

    return result


def compute_min_dcf(fr_rate, fa_rate, p_target=0.05, c_miss=1.0, c_fa=1.0):
    """ Compute normalized minimum detection cost function (minDCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers

    Parameters:
        fr_rate(np.ndarray): the false rejection rate as a function of increasing thresholds.
        fa_rate(np.ndarray): the false acceptance rate as a function of increasing thresholds.
        p_target(float): the prior probability of being a target.
        c_miss(float): cost of miss detection(false rejects).
        c_fa(float): cost of miss detection(false accepts).

    Returns:
        min_cdf(float): the normalized minimum detection cost function (minDCF)

     """

    dcf = c_miss * fr_rate * p_target + c_fa * fa_rate * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_cdf = c_det / c_def
    return min_cdf

def get_precision_reall(scores,labels,th_list):
    # compute precision and recall when threshold is in th_list
    precision_list = []
    recall_list = []



    for th in th_list:
        # get the prediction
        pred = scores > th
        # get tp fp tn fn
        tp = np.sum(pred[labels==1])
        fp = np.sum(pred[labels==0])
        tn = np.sum(~pred[labels==0])
        fn = np.sum(~pred[labels==1])
        # print(f'tp:{tp},fp:{fp},tn:{tn},fn:{fn}')
        # print(f"Total:#{tp+fp+tn+fn}")
        # compute the precision
        precision = np.sum(pred[labels==1]) / np.sum(pred)
        # compute the recall
        recall = np.sum(pred[labels==1]) / np.sum(labels)
        precision_list.append(precision)
        recall_list.append(recall)
    # print in format
    return_string = 'threshold\t\tprecision\t\trecall'
    print('threshold\t\tprecision\t\trecall')
    for th, precision, recall in zip(th_list, precision_list, recall_list):
        print(f'{th:.4f}\t\t{precision*100:.2f}%\t\t{recall*100:.2f}%')
        return_string+=f'{th:.4f}\t\t{precision*100:.2f}%\t\t{recall*100:.2f}\n'
    return return_string
