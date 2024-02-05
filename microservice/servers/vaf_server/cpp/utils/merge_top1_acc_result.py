# coding = utf-8
# @Time    : 2023-05-15  17:46:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 合并所有c++输出结果，统计完整的top1_acc.

import os
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--root_path', type=str, default='./a_split',help='')
parser.add_argument('--save_path', type=str, default='./a_split.csv',help='')
args = parser.parse_args()

def merge_all_txt(rootpath,savepath="./df.csv"):
    """遍历rootpath下所有的\d_results文件夹，分别读取TP_<th>.txt,TN_<th>.txt,FP_<th>.txt,FN_<th>.txt
    将相同的th的结果合并到四个大文件中:TP_<th>.txt,TN_<th>.txt,FP_<th>.txt,FN_<th>.txt
    并获取这四个文件的行数，即得到该阈值下的TP,TN,FP,FN
    计算该阈值下的ACC,Recall,Precision,F1,TNR,TPR,FPR,FNR.
    完成所有阈值遍历后最终数据一个df表格，记录每个阈值下的ACC,Recall,Precision,F1,TNR,TPR,FPR,FNR.
    同时返回每个阈值下的TP,TN,FP,FN的df表格.
    Args:
        rootpath (_type_): 保存分割后结果的文件夹 (1_results,2_results,....,n_results)
    """
    subdirs = [d for d in os.listdir(rootpath) if (os.path.isdir(os.path.join(rootpath, d)) and d.endswith("_results"))]
    subdirs = sorted(subdirs, key=lambda x: int(x.split("_")[0]))
    print(f"All subdirs: {subdirs}")
    th_list = sorted([str(d.split("_")[1].replace(".txt","")) for d in os.listdir(os.path.join(rootpath,subdirs[0])) if d.startswith("TP_")])
    print(f"All th: {th_list}")
    column_names = ["Threshod", "Accuracy", "Recall", "Precision", "F1-Score", "TNR", "TPR", "FPR", "FNR","TN","TP","FN","FP","Total"]
    results_df = pd.DataFrame(columns=column_names)
    for th  in th_list:
        # 合并处理所有阈值的结果
        tp,tn,fp,fn = 1,1,1,1
        for subdir in subdirs:
            tp_file = os.path.join(rootpath, subdir, f"TP_{th}.txt")
            tn_file = os.path.join(rootpath, subdir, f"TN_{th}.txt")
            fp_file = os.path.join(rootpath, subdir, f"FP_{th}.txt")
            fn_file = os.path.join(rootpath, subdir, f"FN_{th}.txt")

            # 读取 TP.txt 文件
            try:
                # check if tp_file exists, if not, continue, tp_lines = []
                if not os.path.exists(tp_file):
                    tp_lines = []
                else:
                    with open(tp_file, 'r') as f:
                        tp_lines = f.readlines()
            except Exception as e:
                print("!!!!!!!!Error!!!!!!!!"*2)
                print(e)
                print("!!!!!!!!Error!!!!!!!!"*2)
                continue

            # 读取 TN.txt 文件
            try:
                if not os.path.exists(tn_file):
                    tn_lines = []
                else:
                    with open(tn_file, 'r') as f:
                        tn_lines = f.readlines()
            except Exception as e:
                print("!!!!!!!!Error!!!!!!!!"*2)
                print(e)
                print("!!!!!!!!Error!!!!!!!!"*2)
                continue

            # 读取 FP.txt 文件
            try:
                if not os.path.exists(fp_file):
                    fp_lines = []
                else:
                    with open(fp_file, 'r') as f:
                        fp_lines = f.readlines()
            except Exception as e:
                print("!!!!!!!!Error!!!!!!!!"*2)
                print(e)
                print("!!!!!!!!Error!!!!!!!!"*2)
                continue

            # 读取 FN.txt 文件
            try:
                if not os.path.exists(fn_file):
                    fn_lines = []
                else:
                    with open(fn_file, 'r') as f:
                        fn_lines = f.readlines()
            except Exception as e:
                print("!!!!!!!!Error!!!!!!!!"*2)
                print(e)
                print("!!!!!!!!Error!!!!!!!!"*2)
                continue

            # for i in range(len(tp_lines)):
            #     tp_data = tp_lines[i].strip().split(",")
            #     tn_data = tn_lines[i].strip().split(",")
            #     fp_data = fp_lines[i].strip().split(",")
            #     fn_data = fn_lines[i].strip().split(",")

            tp += len(tp_lines)
            tn += len(tn_lines)
            fp += len(fp_lines)
            fn += len(fn_lines)
        print(f"Threshod: {th}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * precision * recall / (precision + recall)
        tnr = tn / (tn + fp)
        tpr = recall
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)

        result = {"Threshod": float(th), "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-Score": f1_score, "TNR": tnr, "TPR": tpr, "FPR": fpr, "FNR": fnr, "TN": tn, "TP": tp, "FN": fn, "FP": fp, "Total": tn + tp + fn + fp}
        results_df = results_df.append(result, ignore_index=True)
    if savepath:
        results_df.to_csv(savepath, index=False)
    return results_df


if __name__ == '__main__':
    # python merge_top1_acc_result.py --root_path ./a_split --save_path ./a_split.csv
    merge_all_txt(args.root_path, args.save_path) 