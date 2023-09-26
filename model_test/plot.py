import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
# Set up the command-line argument parser
parser = argparse.ArgumentParser(description='Plot bar charts of EER_mean for different models at each time.')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--png_path', type=str, help='Path to save the PNG file')
args = parser.parse_args()

# if "," in args.csv_path: read all csv and cat them to one csv
if "," in args.csv_path:
    csv_paths = args.csv_path.split(",")
    dfs = [pd.read_csv(csv_path) for csv_path in csv_paths]
    df = pd.concat(dfs)
else:
    df = pd.read_csv(args.csv_path)

# make output's folder
output_folder = os.path.dirname(args.png_path)
print(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the unique models
models = df['model'].unique()

# Set up the figure and axes
fig, ax = plt.subplots()

# Calculate the width of each bar
bar_width = 0.2

# Set the x positions for the bars
x_positions = range(len(models))
time_set = df['time'].unique()
for time_length in time_set:
    time_data = df[df['time'] == time_length]
    print(time_data)
        
    models = time_data['model'].unique()
    trials = time_data['trails'].unique()
    EER_mean = []
    for model in models:
        model_data = time_data[time_data['model'] == model]
        EER_mean.append(model_data['EER_mean'].values)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(models))]

    # 绘制柱状图
    width = 0.15
    x = np.arange(len(trials))

    fig, ax = plt.subplots(figsize=(10, 6),dpi=150)
    for i, model in enumerate(models):
        # if "_and_" in model : replace_str = "_and_" to " & " and add "//" in bar's label
        if "_and_" in model:
            model = model.replace("_and_", " & ")
            hatch = "//"
        else:
            hatch = ""
        ax.bar(x + (i - len(models) / 2) * width, EER_mean[i], width=width, label=model, color=colors[i], hatch=hatch)

    # 添加标签和标题
    ax.set_xlabel(r"$Trials$")
    ax.set_ylabel(r"$Mean\ EER\ (\%)$")
    ax.set_title("Comparison of EER Mean among Different Models (Len: " + str(time_length) + " s)")
    ax.set_xticks(x)
    ax.set_xticklabels(trials)
    ax.legend()
    ax.set_ylabel(r'$Mean\ EER\ (\%)$')
    # Add a legend
    ax.legend(title='Model')
    plt.savefig(args.png_path + str(time_length) + '.png')
    # 显示图形
    plt.show()