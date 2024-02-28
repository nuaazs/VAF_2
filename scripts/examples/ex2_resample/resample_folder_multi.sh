#!/bin/bash

process(){
    tiny_list=$1
    src_folder=$2
    dst_folder=$3
    sample_rate=$4
    file_extension=$5
    channel=$6
    start=$7
    duration=$8
    process_idx=$9
    file_num=$(echo "$tiny_list" | wc -l)
    i=1
    # process each file
    for file in $tiny_list
    do
        # print progress bar, with no newline
        echo -ne "Progress: $i/$file_num\r"
        i=$(($i+1))

        # get dst folder path
        # $1 and $2 are both absolute path, with special characters like  /, *, etc.
        pwdesc1=$(echo $src_folder | sed 's_/_\\/_g')
        pwdesc2=$(echo $dst_folder | sed 's_/_\\/_g')
        # echo $pwdesc1
        # echo $pwdesc2
        dst_file_path=$(echo $file | sed "s/$pwdesc1/$pwdesc2/g")
        dst_folder_path=$(dirname "$dst_file_path")

        # if the destination folder does not exist, create it
        if [ ! -d "$dst_folder_path" ]; then
            # echo "mkdir -p /$dst_folder_path"
            mkdir -p /"$dst_folder_path"
        fi
        
        # check if the file already exists and is larger than 1KB
        if [ -f "$dst_file_path" ] && [ $(stat -c %s "$dst_file_path") -gt 1024 ]; then
            continue
        fi

        # ffmpeg with no output
        # -y: overwrite output file if it exists
        # -i: input file
        # -ac: number of audio channels
        # -ar: audio sample rate
        # -acodec: audio codec
        # -map_channel: map audio channel
        # ffmpeg -y -i "$file" -ac 1 -ar $3 -acodec pcm_s16le -map_channel 0.0.$5 /"$dst_file_path" #> /dev/null 2>&1
        date=$(date +%Y%m%d)
        ffmpeg -y -i "$file" -ac 1 -ar $sample_rate -acodec pcm_s16le -map_channel 0.0.$channel -ss $start -t $duration /"$dst_file_path"  > ${date}_resample.log > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            # write error log
            echo "$file Failed" >> error.log
        fi
    done
    echo -e "\n\nDone!"
}

# resample_folder.sh
# Resample all files in a folder and its subfolders to a given sample rate
# find + ffmpeg
# Usage: resample_folder.sh --i <folder> --o <dst_folder> --sr <sample rate> --ext <file extension> --channel <channel> --start <start> --duration <duration>
# Example: resample_folder.sh --i /home/zs/data/ --o /home/zs/data/resampled/ --sr 16000 --ext wav --channel 0 --start 0 --duration 10

# add --help or -h or --h or -help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: resample_folder.sh <folder> <dst_folder> <sample rate> <file extension> <channel> <start> <duration> <process_num>"
    echo "Example: resample_folder.sh /home/zs/data/ /home/zs/data/resampled/ 16000 wav 0 0 10 4"
    exit 0
fi


# Check if the number of arguments is correct
if [ $# -ne 8 ]; then
    echo "Usage: resample_folder.sh <folder> <dst_folder> <sample rate> <file extension> <channel> <start> <duration> <process_num>"
    exit 1
fi

# Check if the source folder exists
if [ ! -d "$1" ]; then
    echo "Error: source folder does not exist"
    exit 1
fi

# Check if the destination folder exists
# if not exists, create it
if [ ! -d "$2" ]; then
    mkdir -p "$2"
fi

# Check if the sample rate is valid
if [ $3 -ne 16000 ] && [ $3 -ne 48000 ] && [ $3 -ne 8000 ]; then
    echo "Error: sample rate must be 16000, 8000 or 48000"
    exit 1
fi

# Check if the file extension is valid
# file extension must be wav, flac, m4a or mp3
if [ "$4" != "wav" ] && [ "$4" != "flac" ] && [ "$4" != "m4a" ] && [ "$4" != "mp3" ]; then
    echo "Error: file extension must be wav, flac, m4a or mp3"
    exit 1
fi

# Check if the channel is valid
# channel must be 0 or 1
if [ $5 -ne 0 ] && [ $5 -ne 1 ]; then
    echo "Error: channel must be 0 or 1"
    exit 1
fi

# Check if the start is valid
# start must be a positive integer
if [ $6 -lt 0 ]; then
    echo "Error: start must be a positive integer"
    exit 1
fi

# Check if the duration is valid
# duration must be a positive integer
if [ $7 -lt 0 ]; then
    echo "Error: duration must be a positive integer"
    exit 1
fi

# Check if the process_num is valid
# process_num must be a positive integer
if [ $8 -lt 0 ]; then
    echo "Error: process_num must be a positive integer"
    exit 1
fi

# Find all files in the source folder and its subfolders
# and resample them to the given sample rate
# first get file list
# then resample each file

# get file list
file_list=$(find "$1" -type f -name "*.$4" | sed "s_^\./__g")

# echo "$file_list"
file_num=$(echo "$file_list" | wc -l)
echo "Total file num: $file_num"
# echo "Start processing..."
process_num=$8
echo "==========================="
echo ""
echo "Muti Processing #${process_num}.."
echo ""
echo "==========================="
# def a func to process a list of file

# calculate the number of files in each process
files_per_process=$((file_num / process_num))
remainder_files=$((file_num % process_num))

process_idx=1
start_idx=1
for ((i=1; i<=process_num; i++))
do
    end_idx=$((start_idx + files_per_process - 1))
    
    # distribute remainder files evenly to processes
    if [ $i -le $remainder_files ]; then
        end_idx=$((end_idx + 1))
    fi
    
    file_list_tiny=$(echo "$file_list" | sed -n "$start_idx,$end_idx p")
    
    # echo "$file_list_tiny"
    echo "#Process $process_idx: $(echo "$file_list_tiny" | wc -l) files"
    process "$file_list_tiny" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$process_idx" &
    
    start_idx=$((end_idx + 1))
    process_idx=$((process_idx + 1))
done

wait

echo "All processes completed!"

# Check if any files failed to process and capture error logs
failed_files=0

for file in $file_list
do
    dst_file_path=$(echo "$file" | sed "s#$1#$2#g")
    
    if [ ! -f "$dst_file_path" ]; then
        echo "Failed to process file: $file" >> error_logs.txt
        failed_files=$((failed_files+1))
    fi
done

if [ $failed_files -gt 0 ]; then
    echo "Total files failed to process: $failed_files"
    echo "Error logs are written to error_logs.txt"
else
    echo "All files processed successfully."
fi
