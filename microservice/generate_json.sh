#!/bin/bash

# 输入文件路径
input_file="read.txt"
# 输出文件路径
output_file="output.json"

# 创建一个空字典
echo "{" > "$output_file"

# 逐行读取输入文件
while IFS= read -r line || [[ -n "$line" ]]; do
    # 计算文本长度
    length=${#line}
    
    # 跳过小于50个字的文本和空行
    if [ "$length" -ge 200 ] && [ -n "$line" ]; then
        # 将文本内容转义为JSON格式
        escaped_line=$(printf '%s' "$line" | sed 's/"/\\"/g')
        # 在输出文件中写入key-value对
        echo "  \"$((++i))\": \"$escaped_line\"," >> "$output_file"
    fi
done < "$input_file"

# 删除最后一行多余的逗号
sed -i '$ s/,$//' "$output_file"

# 完成json文件
echo "}" >> "$output_file"

echo "生成的JSON文件保存在$output_file"
