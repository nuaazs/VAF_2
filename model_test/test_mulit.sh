#!/bin/bash/

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <conf_directory>"
    exit 1
fi

# 获取配置目录
conf_dir="$1"

# 检查目录是否存在
if [ ! -d "$conf_dir" ]; then
    echo "Directory $conf_dir does not exist"
    exit 1
fi

# 循环处理目录下的所有info文件
for file in "$conf_dir"/*.info; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        bash test.sh -c "$file"
    fi
done