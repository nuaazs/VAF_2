#!/bin/bash
# 同步 /home/zhaosheng/bert_fraud_classify/dguard_nlp
# 到当前目录，除了*.ckpt文件和*.pt文件和*.csv文件，不同步所有大于1MB的文件
# 用法：sh sync.sh
# 作者：zhaosheng

dir="/home/zhaosheng/bert_fraud_classify/dguard_nlp"
echo "同步 $dir 到当前目录，除了*.ckpt文件和*.pt文件和*.csv文件，不同步所有大于1MB的文件"
dest_dir="."
echo "目标目录：$dest_dir"
echo "同步中..."
rsync -av --exclude="*.ckpt" --exclude="*.pt" --exclude="*.csv" --max-size=1m $dir $dest_dir
echo "同步完成"