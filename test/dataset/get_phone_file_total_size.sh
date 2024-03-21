#!/bin/bash
# step 1
## Remove small files


## step 2
## Remove empty folder


## step 3
## Calc Phone size
# /home/zhaosheng/Documents/cjsd_train_data/data_vad_split
# Get each sub folder(folder name is phone num) in PATH and output filepath_size.csv
# <phone>,<size>
# Sorted by size DESC
PATH="data/cti_v2_plus_split"
OUTPUT_FILE="cti_v2_plus_size.csv"

# Function to calculate folder size
get_folder_size() {
    /usr/bin/du -sb "$1" | /usr/bin/awk '{print $1}'
}

# Get sub-folders (phone numbers) in PATH
phone_folders=$(/usr/bin/find "$PATH" -mindepth 1 -maxdepth 1 -type d)

# Iterate through each sub-folder and calculate size
for folder in $phone_folders; do
    size=$(get_folder_size "$folder")
    /usr/bin/echo "$(/usr/bin/basename "$folder"),$size" >> "$OUTPUT_FILE"
done

# Sort by size in descending order
/usr/bin/sort -t ',' -k2,2nr -o "$OUTPUT_FILE" "$OUTPUT_FILE"
