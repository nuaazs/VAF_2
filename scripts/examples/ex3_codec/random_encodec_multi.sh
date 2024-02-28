#!/bin/bash

process(){
    tiny_list=$1
    src_folder=$2
    dst_folder=$3
    process_idx=$4
    file_num=$(echo "$tiny_list" | wc -l)
    i=1
    # process each file
    for file in $tiny_list
    do
        # print progress bar, with no newline
        echo -e "#${process_idx} Progress: $i/$file_num\r"
        i=$(($i+1))

        # echo "Raw file path: $file"

        # get dst folder path
        # $1 and $2 are both absolute path, with special characters like  /, *, etc.
        pwdesc1=$(echo $src_folder | sed 's_/_\\/_g')
        pwdesc2=$(echo $dst_folder | sed 's_/_\\/_g')
        # echo $pwdesc1
        # echo $pwdesc2
        dst_file_path=$(echo $file | sed "s/$pwdesc1/$pwdesc2/g")
        dst_folder_path=$(dirname "$dst_file_path")

        # check if the destination file already exists and its size is larger than 1kb
        if [ -f "$dst_file_path" ] && [ $(stat -c%s "$dst_file_path") -gt 1024 ]; then
            # echo "File $dst_file_path already exists and its size is larger than 1kb. Skipping..."
            continue
        fi

        # if the destination folder does not exist, create it
        if [ ! -d "$dst_folder_path" ]; then
            # echo "mkdir -p $dst_folder_path"
            mkdir -p "$dst_folder_path"
        fi
        
        # def list in shell ogg_li = ['4.5k', '5.5k', '7.7k', '9.5k', '12.5k', '16.0k', '32k']
        ogg_li=('4.5k' '5.5k' '7.7k' '9.5k' '12.5k' '16.0k' '32k') # 'skip'

        # high 4.5k 5.5k 7.7k
        # mid 9.5k
        # low 12.5k 16.0k 32k

        size=${#ogg_li[@]}
        index=$(($RANDOM % $size))
        ogg_bitrate=${ogg_li[$index]}
        
        # echo "Random choose $ogg_bitrate"
        # if = skip, then continue
        if [ "$ogg_bitrate" = "skip" ]; then
            continue
        fi


        # echo $dst_file_path
        # echo "Preprocessed file path: $file"

        # encode, with no nomal output, only error output
        date=$(date +%Y%m%d)
        ffmpeg -y -i "$file" -c:a libopus -b:a "$ogg_bitrate" "$dst_file_path".ogg > ${date}_encodec.log 2>&1 && \

        # decode , with no output
        ffmpeg -y -i "$dst_file_path".ogg -ar 8000 "$dst_file_path" > ${date}_encodec.log 2>&1 && \

        # remove the ogg file
        rm "$dst_file_path".ogg
    done
    echo -e "${process_idx}\n\nDone!"

}



# add --help or -h or --h or -help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: random_encodec.sh <folder> <dst_folder> <file extension> <process_num>"
    echo "Example: random_encodec.sh /home/zs/data/ /home/zs/data/resampled/ wav 4"
    exit 0
fi

# Check if the number of arguments is correct
if [ $# -ne 4 ]; then
    echo "Usage: random_encodec.sh <folder> <dst_folder> <file extension> <process_num>"
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


# Check if the file extension is valid
# file extension must be wav, flac, m4a or mp3
if [ "$3" != "wav" ] && [ "$3" != "flac" ] && [ "$3" != "m4a" ] && [ "$3" != "mp3" ]; then
    echo "Error: file extension must be wav, flac, m4a or mp3"
    exit 1
fi

# Check if the process_num is valid
# process_num must be a positive integer
if ! [[ "$4" =~ ^[0-9]+$ ]]; then
    echo "Error: process_num must be a positive integer"
    exit 1
fi

process_num=$4


# use find to get file list
# then use while loop to process each file

# get file list
file_list=$(find "$1" -type f -name "*.$3" | sed "s_^\./__g")
# echo "$file_list"
# total file num
file_num=$(echo "$file_list" | wc -l)

# add progress bar
echo "Total file num: $file_num"
echo "Start processing..."

# sort file list
file_list=$(echo "$file_list" | sort)

# split file_list to #process_num parts, and process each part in parallel

# split file_list to #process_num parts
length_tiny=$(( $file_num / $process_num ))
remainder=$(( $file_num % $process_num ))
start_index=1

# for process_idx in $(seq 1 $process_num)
# do
#     if [ $process_idx -le $remainder ]; then
#         end_index=$(( $start_index + $length_tiny ))
#     else
#         end_index=$(( $start_index + $length_tiny - 1 ))
#     fi

#     # get tiny list by start_index and end_index
#     file_list_tiny=$(echo "$file_list" | awk "NR>=$start_index && NR<=$end_index")
#     start_index=$(( $end_index + 1 ))

#     echo "#Process $process_idx: $(echo "$file_list_tiny" | wc -l) files"
#     process "$file_list_tiny" "$1" "$2" "$process_idx" &
# done

# wait

echo "All processes completed."

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
