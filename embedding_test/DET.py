import argparse
import pandas as pd
import matplotlib.pyplot as plt

def compute(data):
    thresholds = data['Threshod'].tolist()
    fn = data['FN'].tolist()
    fp = data['FP'].tolist()
    tn = data['TN'].tolist()
    tp = data['TP'].tolist()

    fpr = [fp[i] / (fp[i] + tn[i]) for i in range(len(thresholds))]
    fnr = [fn[i] / (fn[i] + tp[i]) for i in range(len(thresholds))]

    # 计算data1 EER点
    min_diff = float('inf')
    for i in range(len(thresholds)):
        diff = abs(fpr[i] - fnr[i])
        if diff < min_diff:
            min_diff = diff
            eer_idx1 = i
    return fpr,fnr,eer_idx1


def plot_det_curve(data_list,label_list, output_img):
    # 读取data1
    color_list = ['b','g','r','pink','blue']
    
    plt.figure(figsize=(8, 8))
    for _index,data in enumerate(data_list):
        color = color_list[_index%len(color_list)]
        fpr,fnr,eer_idx = compute(data)
    
        plt.plot(fpr, fnr, color=color, label=label_list[_index]) 
        plt.scatter(fpr[eer_idx], fnr[eer_idx], s=50, color=color)


    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.2)
    plt.title('DET Curve')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, required=True, help='path to csv files, split by ,')
    parser.add_argument('--labels', type=str, required=True, help='path to csv files, split by ,')
    parser.add_argument('--output', type=str, required=True, help='output image path')
    args = parser.parse_args()
    
    data_file_paths = args.files.split(",")
    labels = args.labels.split(",")
    
    data = [pd.read_csv(csv_file) for csv_file in data_file_paths] 
    
    plot_det_curve(data,labels, args.output)
