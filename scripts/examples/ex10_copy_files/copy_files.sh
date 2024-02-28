#!/bin/bash

# 路径参数
path1="/datasets/common_voice_kpd" # 原路径
path2="/datasets/common_voice_kpd_phone" # 目标路径
ext=".wav"    # 排除的文件扩展名

# 复制目录结构
rsync -av --exclude="*$ext" "$path1"/ "$path2"/

# 提示复制完成
echo "复制完成"
