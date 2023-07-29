#!/bin/bash

# source_dir="/home/zhaosheng/3D-Speaker/dguard"
# target_dir="/VAF/train/dguard"

# source_dir2="/home/zhaosheng/3D-Speaker/egs"
# target_dir2="/VAF/train/egs"

source_dir3="/home/duanyibo/dyb/test_model"
target_dir3="/VAF/test"

sync_directories() {
    # rsync -av --delete "$1/" "$2"
    rsync -av --delete --exclude=".*" --exclude="result" --exclude="*.npy" --exclude="*.ark" --exclude="*data*" --exclude="*CKPT*" "$1/" "$2"
}

cron_expr="0 0 * * *"

# mkdir -p "$target_dir"
mkdir -p "$target_dir2"
mkdir -p "$target_dir3"

sync_directories "$source_dir2" "$target_dir2"
sync_directories "$source_dir3" "$target_dir3"

# (crontab -l 2>/dev/null; echo "$cron_expr cd $source_dir && $0") | crontab -

