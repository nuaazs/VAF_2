import os
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--score_path', type=str, default=" ",help='')
parser.add_argument('--save_scores', type=str, default="0",help='GPU')
args = parser.parse_args()
for i in [2279]:
    with open(args.score_path,"r") as f:
        lines = f.readlines()
    lines.sort(key=lambda x: float(x.split(",")[2]),reverse=True)

    with open(args.save_scores,'w') as filesort:
        filesort.writelines(lines)


