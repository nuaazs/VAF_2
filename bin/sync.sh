#!/bin/bash

# 设置源目录和目标目录
# source_dir="/home/zhaosheng/3D-Speaker/dguard"
# target_dir="/VAF/train/dguard"

source_dir2="/home/zhaosheng/3D-Speaker/egs"
target_dir2="/VAF/train/egs"

source_dir3="/home/duanyibo/dyb/test_model"
target_dir3="/VAF/test"

# 同步函数
sync_directories() {
    # rsync -av --delete "$1/" "$2"
    # 同步时排除所有隐藏文件及文件名中包含data或者CKPT的文件夹
    rsync -av --delete --exclude=".*" --exclude="result" --exclude="*.npy" --exclude="*.ark" --exclude="*data*" --exclude="*CKPT*" "$1/" "$2"
}

# 设置定时任务，每天执行一次同步操作（可根据需求调整）
cron_expr="0 0 * * *"

# 检查目标目录是否存在，若不存在则创建
# mkdir -p "$target_dir"
mkdir -p "$target_dir2"
mkdir -p "$target_dir3"

# 执行一次初始同步
# sync_directories "$source_dir" "$target_dir"
sync_directories "$source_dir2" "$target_dir2"
sync_directories "$source_dir3" "$target_dir3"

# 添加定时任务到crontab
# (crontab -l 2>/dev/null; echo "$cron_expr cd $source_dir && $0") | crontab -

# echo "已设置定时同步任务！"