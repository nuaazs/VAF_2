#!/bin/bash
# models="repvgg eres2net dfresnet_233"
exp="resnet152_0912"
models="resnet152_lm"
MD_save_path="/duanyibo/maidian/data/test_cti-v2/"
ZC_save_path="/duanyibo/maidian/data/enroll_cti-v2_3w/"

# enroll.bin  enroll.txt
# ====================================================================================
calc_thread=24
th_start=0.10
th_stop=0.95
th_step=0.01
SAVE_DIR="/result"
REMOVE="no"

# ====================================================================================
mkdir -p $SAVE_DIR
mkdir -p ./log
for model in $models;do
	GR_save_path="/result/$exp/bin/$model"
	CS_save_path="/result/$exp/CS_bin/$model"
	if [ $REMOVE == "yes" ] && [ $exp != "" ]  && [ $model != "" ]; then
		echo "** Remove PATH: ${CS_save_path} "
		rm -rf $CS_save_path
	fi
        mkdir -p $CS_save_path
	echo "* CS(MD+GR) path: $CS_save_path"
	echo "* GR        path: $GR_save_path"
	echo "* MD        path: $MD_save_path"
	echo "* ZC        path: $ZC_save_path"
	EMB_SIZE=256
	if [ $model == "dfresnet_233" ]; then
	    EMB_SIZE=512
	elif [ $model == "repvgg" ]; then  
	    EMB_SIZE=512
	elif [ $model == "eres2net" ]; then
	    EMB_SIZE=192
	fi
	# MD @longyuan
	MD_txt=$MD_save_path/$model/enroll.txt
	MD_bin=$MD_save_path/$model/enroll.bin

	GR_bin=$GR_save_path/enroll.bin
	GR_txt=$GR_save_path/enroll.txt

	# ZC @longyuan
	ZC_bin=$ZC_save_path/$model/enroll.bin
	ZC_txt=$ZC_save_path/$model/enroll.txt
	
	CS_bin=$CS_save_path/enroll.bin
	CS_txt=$CS_save_path/enroll.txt

	python add_bin.py --GR_path $GR_bin --MD_path $MD_bin --save_path $CS_bin > ./log/add_bin.out
	cat $GR_txt $MD_txt >$CS_txt
	a_split_dir=$SAVE_DIR/$exp/acc_splits/${model}
	acc_save_dir=$SAVE_DIR/$exp/acc_results/${model}

	echo "All Bin Okay! Now Split ..."
	if [ $REMOVE == "yes" ] && [ $exp != "" ]  && [ $model != "" ]; then
		echo "** Remove PATH: ${a_split_dir} "
		rm -rf $a_split_dir
	fi
	if [ $REMOVE == "yes" ] && [ $exp != "" ]  && [ $model != "" ]; then
		echo "** Remove PATH: ${acc_save_dir} "
		rm -rf $acc_save_dir
	fi

	mkdir -p $a_split_dir
	mkdir -p $acc_save_dir 
        python utils/split_vector.py --raw_bin_path $CS_bin --raw_txt_path $CS_txt --number $calc_thread --emb_size $EMB_SIZE --save_folder $a_split_dir > ./log/split_vector.log
	echo "Splited Data Save to -> $a_split_dir"

	for file_num in $(seq 0 $(($calc_thread-1))); do
		echo "file_num: $file_num"
		CS_txt_tiny=$a_split_dir/id_${file_num}.txt
		CS_bin_tiny=$a_split_dir/vector_${file_num}.bin
		score_split=$a_split_dir/score/$file_num.score
		result_split=$a_split_dir/result/${file_num}_results
		mkdir -p $a_split_dir/score $a_split_dir/result
		CS_len_tiny=$(cat $CS_txt_tiny | wc -l)
		ZC_len=$(cat $ZC_txt | wc -l)
		echo "Find TOP1 => A: $CS_len_tiny B: $ZC_len EMB size: $EMB_SIZE"
		echo "    A bin: $CS_bin_tiny"
		echo "    B bin: $ZC_bin"
		echo "    A txt: $CS_txt_tiny"
		echo "    B txt: $ZC_txt"
		echo "     => SAVE TO: $score_split"
		echo ""
		echo "CMD: utils/top1 $CS_len_tiny $ZC_len $EMB_SIZE $CS_bin_tiny $ZC_bin $CS_txt_tiny $ZC_txt $score_split"
		utils/top1 $CS_len_tiny $ZC_len $EMB_SIZE $CS_bin_tiny $ZC_bin $CS_txt_tiny $ZC_txt $score_split & > ./log/top1_${file_num}.out
	done
	wait
	for file_num in $(seq 0 $(($calc_thread-1))); do
		result_split=$a_split_dir/result/${file_num}_results
		score_split=$a_split_dir/score/$file_num.score
	        echo "Get TOP1 ACC => $result_split"	
		utils/top1acc $score_split $th_start $th_stop $th_step $result_split & > ./log/top1acc_${file_num}.out
	done
	wait
	echo "Merge Result ..."
	python utils/merge_top1_acc_result.py --root_path $a_split_dir/result --save_path $acc_save_dir/result.csv
	cat $a_split_dir/score/*.score >$a_split_dir/scores.score
        python top700.py --score_path $a_split_dir/scores.score --save_scores $acc_save_dir/sort.score
done

