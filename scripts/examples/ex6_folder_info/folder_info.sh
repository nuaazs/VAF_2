#!/bin/bash
# coding = utf-8
# @Time    : 2023-03-15  20:12:38
# @Author  : zhaosheng@nuaa.edu.cn

# Get all information of a folder
# Usage: folder_info.sh <folder> <file extension>
# Example: folder_info.sh /home/zs/data/ wav

# add help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: folder_info.sh <folder> <file extension>"
    echo "Example: folder_info.sh /home/zs/data/ wav"
    exit 0
fi

# Check if the number of arguments is correct
if [ $# -ne 2 ]; then
    echo "Usage: folder_info.sh <folder> <file extension>"
    exit 1
fi

# Check if the source folder exists
if [ ! -d "$1" ]; then
    echo "Error: source folder does not exist"
    exit 1
fi

# Get info
echo "Folder: $1"
echo "File extension: $2"

# Get audio total duration
# use find to get file list
# then use while loop to process each file
total_duration=0
total_size=0
# get file list
file_list=$(find "$1" -type f -name "*.$2" | sed "s_^\./__g")
file_num=$(echo "$file_list" | wc -l)
echo "Total file num: $file_num"
i=1
# process each file

for file in $file_list
do
    # get subfolder name
    subfolder=$(dirname "$file")
    subfolder=$(dirname "$subfolder")
    id=$(basename "$subfolder")
    # echo "Subfolder: $subfolder"
    
    


    # add progress bar
    echo -ne "Progress: $i/$file_num\r"
    i=$(($i+1))

    # get duration
    duration=$(ffprobe -i "$file" -show_entries format=duration -v quiet -of csv="p=0")
    # conver to float
    duration=$(echo "$duration" | awk '{printf "%.0f", $1}')
    # add duration
    # total_duration+=$duration
    total_duration=$(echo "$total_duration+$duration" | bc)
    
    # get file size in bytes
    size=$(stat -c%s "$file")
    # conver to float
    size=$(echo "$size" | awk '{printf "%.0f", $1}')
    # size=0
    # total_size+=$size
    total_size=$(echo "$total_size+$size" | bc)

    # echo "File: $file Duration: $duration seconds Size: $size bytes"
done
# echo "Total size: $total_size bytes"

echo "================================================="
# conver size to MB
total_size=$(echo "$total_size/1024/1024" | bc)
echo "Total size: $total_size MB"
# echo average size with float MB， with 2 decimal places
average_size=$(echo "$total_size/$file_num" | bc -l)
# change to .2float
average_size=$(echo "$average_size" | awk '{printf "%.2f", $1}')
echo "Average size: $average_size MB"

# convert to hour
total_duration_h=$(echo "$total_duration/3600" | bc -l)
# change to .2float
total_duration_h=$(echo "$total_duration_h" | awk '{printf "%.2f", $1}')
echo "Total duration: $total_duration seconds"
echo "Total duration: $total_duration_h hours"

average_duration=$(echo "$total_duration/$file_num" | bc -l)
average_duration=$(echo "$average_duration" | awk '{printf "%.2f", $1}')
echo "Average duration: $average_duration seconds"
echo "================================================="
