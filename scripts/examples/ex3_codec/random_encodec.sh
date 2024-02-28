#!/bin/bash

# add --help or -h or --h or -help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: random_encodec.sh <folder> <dst_folder> <file extension>"
    echo "Example: random_encodec.sh /home/zs/data/ /home/zs/data/resampled/ wav"
    exit 0
fi

# Check if the number of arguments is correct
if [ $# -ne 3 ]; then
    echo "Usage: random_encodec.sh <folder> <dst_folder> <file extension>"
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
i=1
# process each file
for file in $file_list
do
    # print progress bar, with no newline
    echo -ne "Progress: $i/$file_num\r"
    i=$(($i+1))

    # echo "Raw file path: $file"

    # get dst folder path
    # $1 and $2 are both absolute path, with special characters like  /, *, etc.
    pwdesc1=$(echo $1 | sed 's_/_\\/_g')
    pwdesc2=$(echo $2 | sed 's_/_\\/_g')
    # echo $pwdesc1
    # echo $pwdesc2
    dst_file_path=$(echo $file | sed "s/$pwdesc1/$pwdesc2/g")
    dst_folder_path=$(dirname "$dst_file_path")

    # if the destination folder does not exist, create it
    if [ ! -d "$dst_folder_path" ]; then
        # echo "mkdir -p $dst_folder_path"
        mkdir -p "$dst_folder_path"
    fi
    
    # def list in shell ogg_li = ['4.5k', '5.5k', '7.7k', '9.5k', '12.5k', '16.0k', '32k']
    ogg_li=('4.5k' '5.5k' '7.7k' '9.5k' '12.5k' '16.0k' '32k' 'skip')

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
echo -e "\n\nDone!"