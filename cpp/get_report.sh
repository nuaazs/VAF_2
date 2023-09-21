#!/bin/bash
# 给定两批数据，数据A:所有底库文件，数据B:所有测试文件

EMB_SIZE=192
api_thread=48
calc_thread=96
# 指定目录地址
a_dir=/datasets_hdd/datasets/cjsd_download # 待处理音频
b_dir=data/dir_b
a_vad_dir=data/dir_a_vad
b_vad_dir=data/dir_a_vad
a_emb_dir=data/emb_a
b_emb_dir=data/emb_b
tempa=data/temp_a
tempb=data/temp_b
inputa=data/input_a
inputb=data/input_b

# # # 1. 对数据A进行预处理
# python get_vad.py --fold_path $a_dir --dst_path $a_vad_dir --thread $api_thread
# # # 2. 对数据B进行预处理
# python get_vad.py --fold_path $b_dir --dst_path $b_vad_dir --thread $api_thread
# # 3. 对预处理后的A进行特征提取
# python get_embedding.py --fold_path $a_vad_dir --dst_path $a_emb_dir --thread $api_thread
# # 4. 对预处理后的B进行特征提取
# python get_embedding.py --fold_path $b_vad_dir --dst_path $b_emb_dir --thread $api_thread

# 5. 二进制
rm -rf ${tempa}/*
rm -rf ${tempb}/*
rm -rf ${inputa}/*
rm -rf ${inputb}/*

python utils/get_vector.py --save_tiny_folder ${tempa} --thread ${calc_thread}
# mv add_to_a to ${tempa}
mv data/add_to_a/* ${tempa}
python utils/merge_vector.py --fold_path ${tempa} --output vector_a_all 
# bin:data/temp_a/vector_a_all.bin  txt:data/temp_a/vector_a_all.txt
# mv to input
mv ${tempa}/vector_a_all.txt ${inputa}
mv ${tempa}/vector_a_all.bin ${inputa}

python utils/get_vector.py --save_tiny_folder ${tempb} --thread ${calc_thread}
# mv add_to_b to ${tempb}
mv data/add_to_b/* ${tempb}
python utils/merge_vector.py --fold_path ${tempb} --output vector_b_all
# bin:data/temp_b/vector_b_all.bin  txt:data/temp_b/vector_b_all.txt
# mv to input
mv ${tempb}/vector_b_all.txt data/input_b
mv ${tempb}/vector_b_all.bin data/input_b

# 7. split a.bin a.txt为多个小文件,格式 1.bin 1.txt ...
python utils/merge_vector.py --fold_path ${inputa} --output vector_a_final
python utils/merge_vector.py --fold_path ${inputb} --output vector_b_final
python utils/split_vector.py --raw_bin_path ${inputa}/vector_a_final.bin --raw_txt_path ${inputa}/vector_a_final.txt --number ${calc_thread} --save_folder ${inputa}/vector_a_all_split_data

# 8.计算分数
b_txt_path="${inputb}/vector_b_final.txt"
b_bin_path="${inputb}/vector_b_final.bin"
a_split_dir="data/input_a/vector_a_all_split_data"
b_len=$(cat $b_txt_path | wc -l)
for file_num in $(seq 0 $((calc_thread-1)))
do
    echo "file_num: $file_num"
    # 获取txt文件和bin文件地址
    txt_path=${a_split_dir}/id_${file_num}.txt
    bin_path=${a_split_dir}/vector_${file_num}.bin
    # a的长度为txt文件的行数
    a_len=$(cat ${txt_path} | wc -l)
    # echo "utils/top1 $a_len $b_len $EMB_SIZE $bin_path $b_bin_path $txt_path $b_txt_path $a_split_dir/$file_num.score"
    utils/top1 $a_len $b_len $EMB_SIZE $bin_path $b_bin_path $txt_path $b_txt_path $a_split_dir/$file_num.score &
done
wait
echo "Done"

# 9. 统计
for file_num in $(seq 0 $((calc_thread-1)))
do
    utils/top1acc ${a_split_dir}/${file_num}.score 0.1 1.05 0.05 ${a_split_dir}/${file_num}_results &
done
wait
echo "Done"

# 10.合并结果
a_csv_path=data/result.csv
python utils/merge_top1_acc_result.py --root_path $a_split_dir --save_path ${a_csv_path}