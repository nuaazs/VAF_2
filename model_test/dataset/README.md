长江时代训练数据集制作

```shell
# Step 1 Get wav.scp
bash get_wav_list.sh
```



```shell
# Pipeline
data_root="/datasets/cjsd_download"
dataset_name="cti_v2_plus"

# Step 1 Get wav.scp
find $data_root -name "*.wav" > $dataset_name.scp

# Step 2 Get dataset_name.txt
# <phone>,<fid>,<filepath>
for line in $(cat $dataset_name.scp); do
    fid=$(basename $line .wav)
    # phone=line.split("/")[-2]
    phone=$(echo $line | awk -F '/' '{print $(NF-1)}')
    echo "$phone,$fid,$line"
done > $dataset_name.txt


# Step 3 VAD
for line in `cat $dataset_name.txt`
do
    phone=`echo $line | awk -F ',' '{print $1}'`
    fid=`echo $line | awk -F ',' '{print $2}'`
    wav_path=`echo $line | awk -F ',' '{print $3}'`
    mkdir -p data/${dataset_name}_raw/$phone
    mkdir -p data/${dataset_name}_vad/$phone
    mkdir -p data/${dataset_name}_bin/$phone
    mkdir -p data/${dataset_name}_info/$phone
    ffmpeg -i $wav_path -ss 7 data/${dataset_name}_raw/$phone/$fid.wav -ar 16000 -y > /dev/null 2>&1 # -map_channel 0.0.1
    ffmpeg -i data/${dataset_name}_raw/$phone/$fid.wav -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  data/${dataset_name}_bin/$phone/$fid.bin > /dev/null 2>&1
    ./vad_wav --wav-bin=data/${dataset_name}_bin/$phone/$fid.bin --text-out=data/${dataset_name}_info/$phone/$fid.txt --energy-thresh=10e7 --wav-out=data/${dataset_name}_vad/$phone/$fid.wav > /dev/null 2>&1
    # if data/$phone/$fid.wav filesize < 1000, then delete it
    if [ ! -f data/${dataset_name}_vad/$phone/$fid.wav ]; then
        rm -rf data/${dataset_name}_raw/$phone/$fid.wav
        rm -rf data/${dataset_name}_bin/$phone/$fid.bin
        rm -rf data/${dataset_name}_info/$phone/$fid.txt
    fi
done

# Step 4 Split wav
python get_6s_split_result.py --vad_dir=data/${dataset_name}_vad --split_dir=data/${dataset_name}_split --info_dir=data/${dataset_name}_info


# Step 5 Get phone size

OUTPUT_FILE="${dataset_name}_size.csv"

# Function to calculate folder size
get_folder_size() {
    du -sb "$1" | awk '{print $1}'
}

# phone_folders is subfolders of data/${dataset_name}_split
phone_folders=$(find data/${dataset_name}_split -maxdepth 1 -mindepth 1 -type d)
# change \n to space
phone_folders=$(echo "$phone_folders" | tr '\n' ' ')

for folder in $phone_folders; do
    echo "Processing $folder"
    size=$(get_folder_size "$folder")
    echo "$(basename "$folder"),$size" >> "$OUTPUT_FILE"
done

# Sort by size in descending order
sort -t ',' -k2,2nr -o "$OUTPUT_FILE" "$OUTPUT_FILE"

mkdir -p data/${dataset_name}_final

# Step 6 Get final dataset
for line in `cat $OUTPUT_FILE`
do
    phone=`echo $line | awk -F ',' '{print $1}'`
    # if len(phone) == 10 ,add "0"
    if [ ${#phone} -eq 10 ];then
        phone="0$phone"
    fi
    mkdir -p data/${dataset_name}_final/$phone
    cp -r data/${dataset_name}_split/$phone data/${dataset_name}_final

done

#  Step 7 , remove wav files which filesize < 10kb in data/${dataset_name}_final
find data/${dataset_name}_final -name "*.wav" -size -10k -delete
find data/${dataset_name}_final -type d -empty -delete

```