#!/bin/bash
# models="repvgg eres2net dfresnet_233"
exp="repvgg_0914"
models="repvgg"
MD_save_path="/duanyibo/maidian/data/702_bin_path/"
ZC_save_path="/duanyibo/maidian/data/30630_bin_path/"
MD_save_path2="/duanyibo/maidian/data/702_bin_path/"
ZC_save_path2="/duanyibo/maidian/data/30630_bin_path/"

# enroll.bin  enroll.txt
# ====================================================================================
calc_thread=24
th_start=0.10
th_stop=0.95
th_step=0.01
SAVE_DIR="/result"
REMOVE="no"

# ====================================================================================
mkdir -p ./log_fusion
for model in $models;do
	a_split_dir=/result/resnet101_cjsd_0926/acc_splits_702/resnet101_lm
	b_split_dir=/result/resnet221_cjsd_1001/acc_splits_702/resnet221_lm
	c_split_dir=/result/repvgg_0914/acc_splits/repvgg
	fusion_split_dir=/result/fusion_107/221_repvgg
	mkdir -p $fusion_split_dir
	for file_num in $(seq 0 $(($calc_thread-1))); do
		echo "file_num: $file_num"
		# model1
		CS_txt_tiny_a=$a_split_dir/id_${file_num}.txt
		CS_bin_tiny_a=$a_split_dir/vector_${file_num}.bin
		# model2
		CS_txt_tiny_b=$b_split_dir/id_${file_num}.txt
                CS_bin_tiny_b=$b_split_dir/vector_${file_num}.bin
		# model3
		CS_txt_tiny_c=$c_split_dir/id_${file_num}.txt
                CS_bin_tiny_c=$c_split_dir/vector_${file_num}.bin

		# score_fusion
		score_split=$fusion_split_dir/score_fusion/$file_num.score
		result_split=$fusion_split_dir/result_fusion/${file_num}_results
		mkdir -p $fusion_split_dir/score_fusion $fusion_split_dir/result_fusion
		
			
		#echo "utils/top1_mutil_model \
		#	256,256 \
		#	$CS_bin_tiny_a,$CS_bin_tiny_b \
		#	/duanyibo/maidian/data/927/3w_88_bin_path/resnet101_lm/enroll.bin,/duanyibo/maidian/data/106/3w_88_bin_path/resnet221_lm/enroll.bin \
		#	$CS_txt_tiny_a /duanyibo/maidian/data/927/3w_88_bin_path/resnet101_lm/enroll.txt"
		# 1.160w 2.3w+88 3.txt
		# three 

		#utils/top1_mutil_model \
		#	256,256,256 \
		#	$CS_bin_tiny_a,$CS_bin_tiny_b,$CS_bin_tiny_c \
		#	/duanyibo/maidian/data/107/3w_88_bin_path/resnet101_lm/enroll.bin,/duanyibo/maidian/data/107/3w_88_bin_path/resnet221_lm/enroll.bin,/duanyibo/maidian/data/3w_cjsdganrao_maidian_0920/resnet152_lm/enroll.bin \
		#	$CS_txt_tiny_a /duanyibo/maidian/data/3w_cjsdganrao_maidian_0920/resnet152_lm/enroll.txt $score_split & > ./log_fusion/top1_${file_num}.out
		
		utils/top1_mutil_model \
			256,512 \
			$CS_bin_tiny_b,$CS_bin_tiny_c \
			/duanyibo/maidian/data/107/3w_88_bin_path/resnet221_lm/enroll.bin,/duanyibo/maidian/data/3w_cjsdganrao_maidian_0920/repvgg/enroll.bin \
			$CS_txt_tiny_a /duanyibo/maidian/data/107/3w_88_bin_path/resnet101_lm/enroll.txt $score_split & > ./log_fusion/top1_${file_num}.out
	done
	wait
	for file_num in $(seq 0 $(($calc_thread-1))); do
		result_split=$fusion_split_dir/result_fusion/${file_num}_results
		score_split=$fusion_split_dir/score_fusion/$file_num.score
	        echo "Get TOP1 ACC => $result_split"	
		utils/top1acc $score_split $th_start $th_stop $th_step $result_split & > ./log_fusion/top1acc_${file_num}.out
	done
	wait
	echo "Merge Result ..."
	echo " "
	python utils/merge_top1_acc_result.py --root_path $fusion_split_dir/result_fusion --save_path $fusion_split_dir/result.csv
	cat $fusion_split_dir/score_fusion/*.score >$fusion_split_dir/scores.score
        python top700.py --score_path $fusion_split_dir/scores.score --save_scores $fusion_split_dir/sort.score
done

