import pandas as pd
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='Compute mean and standard deviation of CSV files.')
parser.add_argument('--csv_folder', type=str, help='csv_folder')

parser.add_argument('--prefix', type=str, help='prefix')
parser.add_argument('--output', type=str, help='Output file path')


args = parser.parse_args()

csv_files = sorted([os.path.join(args.csv_folder,_file) for _file in os.listdir(args.csv_folder) if _file.split("/")[-1].startswith(args.prefix)])


# 读取所有CSV文件
dfs = [pd.read_csv(file) for file in csv_files]

# 合并数据框
merged_df = pd.concat(dfs)

# 按照'model', 'trails', 'time', 'trial_name'分组，计算平均值和标准差
grouped_df = merged_df.groupby(['model', 'trails', 'time', 'trial_name']).agg({'EER': [np.mean, np.std],
                                                                               'minDCF': [np.mean, np.std]})

# 重命名列名
grouped_df.columns = ['EER_mean', 'EER_std', 'minDCF_mean', 'minDCF_std']

# 重置索引
grouped_df = grouped_df.reset_index()

# 格式化输出列
grouped_df['EER'] = grouped_df.apply(lambda row: f"{row['EER_mean']:.4f}+-{row['EER_std']:.4f}", axis=1)
grouped_df['minDCF'] = grouped_df.apply(lambda row: f"{row['minDCF_mean']:.4f}+-{row['minDCF_std']:.4f}", axis=1)

# 保存结果
grouped_df.to_csv(args.output, index=False)