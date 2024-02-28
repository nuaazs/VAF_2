#!/bin/bash

dir_a="$1"
dir_b="$2"
file_extension="$3"

# 获取文件相对路径
get_relative_path() {
    local absolute_path="$1"
    local base_dir="$2"
    
    echo "${absolute_path#$base_dir/}"
}

# 递归检查目录中的特定后缀文件，并进行重采样
check_files() {
    total_files=0
    missing_files=0

    for file_a in $(find "$1" -type f -name "*.$file_extension"); do
        total_files=$((total_files + 1))
        relative_path=$(get_relative_path "$file_a" "$dir_a")
        file_b="${dir_b}/${relative_path}"
        if [[ ! -e "$file_b" ]]; then
            missing_files=$((missing_files + 1))
            echo "$file_a 存在。"
            echo "$file_b 不存在。"
            # 运行 resample，将 A 目录中的文件通过 ffmpeg 重采样到 8k 并复制到 B 目录下
            # ffmpeg -i "$file_a" -ar 8000 "${file_b}"
        fi
    done

    missing_percent=$(bc <<< "scale=2; ($missing_files / $total_files) * 100")

    echo "总共检查了 $total_files 个文件。"
    echo "$missing_files 个文件不存在，占比 $missing_percent%。"
}

# 检查输入参数个数是否正确
if [[ $# -ne 3 ]]; then
    echo "脚本需要三个参数：目录A，目录B和文件后缀。"
    exit 1
fi

# 检查目录 A 中的文件在目录 B 中是否存在，并进行重采样
check_files "$dir_a"

# 检查目录 B 中的文件在目录 A 中是否存在，并进行重采样
# check_files "$dir_b"
