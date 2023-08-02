#!/bin/bash
set -e
. ./path.sh || exit 1

gpus="1 2 3 " 
#模型选择
model_path="mfa_conformer" #dfresnet_233 mfa_conformer ecapatdnn_1024 repvgg CAMPP_EMB_512 ECAPA_TDNN_1024_EMB_192 ERES2NET_BASE_EMB_192 REPVGG_TINY_A0_EMB_512 DFRESNET56_EMB_512
#测试集选择
trials_class="cnceleb" # voxceleb cnceleb cti 3dspeaker male female
# trials_class="voxceleb"

#测试集数据scp文件地址
vox_scp=/home/duanyibo/dyb/test_model/dataset/voxceleb1/wav.scp
cnceleb_scp=/home/duanyibo/dyb/test_model/dataset/cnceleb_files/eval/wav.scp
cti_scp=/home/duanyibo/dyb/test_model/dataset/cti_test/cti.scp
speaker_scp=/home/duanyibo/dyb/test_model/dataset/3D-speaker/files/wav.scp
male_scp=/datasets/test/testdata_1c_vad_16k/test_scp/male.scp
female_scp=/datasets/test/testdata_1c_vad_16k/test_scp/female.scp


#测试对地址
trials_vox="/home/duanyibo/dyb/test_model/dataset/voxceleb1/trials/vox1_O_cleaned.trial /home/duanyibo/dyb/test_model/dataset/voxceleb1/trials/vox1_E_cleaned.trial /home/duanyibo/dyb/test_model/dataset/voxceleb1/trials/vox1_H_cleaned.trial"
trials_cnceleb=/home/duanyibo/dyb/test_model/dataset/cnceleb_files/eval/trials/CNC-Eval-Avg.lst
trials_cti=/home/duanyibo/dyb/test_model/dataset/cti_test/cti.trials
trials_3dspeaker="/home/duanyibo/dyb/test_model/dataset/3D-speaker/files/trials/trials_cross_device /home/duanyibo/dyb/test_model/dataset/3D-speaker/files/trials/trials_cross_distance /home/duanyibo/dyb/test_model/dataset/3D-speaker/files/trials/trials_cross_dialect"
trials_male=/datasets/test/testdata_1c_vad_16k/test_trials/male.trials
trials_female=/datasets/test/testdata_1c_vad_16k/test_trials/female.trials

#并发数（跟GPU有关，最好为GPU的整数倍）
nj=9

. utils/parse_options.sh || exit 1
#保存结果的地址
result_path=./result

#准备3Dspeaker,voxceleb1,cnceleb1数据
# In this stage we prepare the raw datasets, including Voxceleb1 and Voxceleb2.
# echo "Stage1: Preparing Voxceleb1 3Dspeaker cnceleb1 dataset..."
# ./local/prepare_test_data.sh 

for model_id in $model_path; do
        wechat echo "$model_id start"
        echo "Stage5: Computing $trials_class result"
        
        for trial_class in $trials_class; do
                if [ "$trial_class" == "voxceleb" ]; then
                        echo "voxceleb dataset"
                        if [ ! -d $result_path/$model_id/voxceleb_result/embeddings ]; then
                                torchrun --nproc_per_node=$nj  --master_port=35641 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/voxceleb_result \
                                                                --data $vox_scp --use_gpu --gpu $gpus || wechat echo "voxceleb torchrun error"
                                mkdir -p $result_path/$model_id/voxceleb_result
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/voxceleb_result/embeddings --test_data $result_path/$model_id/voxceleb_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/voxceleb_result/scores --trials $trials_vox || wechat echo "voxceleb compute_score_metrics error"
                fi

                if [ "$trial_class" == 'cnceleb' ]; then
                        echo "cnceleb dataset"
                        if [ ! -d $result_path/$model_id/cnceleb_result/embeddings ]; then
                                torchrun --nproc_per_node=$nj --master_port=45642 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/cnceleb_result \
                                                                --data $cnceleb_scp --use_gpu --gpu $gpus || wechat echo "cnceleb torchrun error"
                                mkdir -p $result_path/$model_id/cnceleb_result 
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/cnceleb_result/embeddings --test_data $result_path/$model_id/cnceleb_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/cnceleb_result/scores --trials $trials_cnceleb || wechat echo "cnceleb compute_score_metrics error"
                fi

                if [ "$trial_class" == 'cti' ]; then
                        echo "cti dataset"
                        if [ ! -d $result_path/$model_id/cti_result/embeddings ]; then
                                torchrun --nproc_per_node=$nj --master_port=55643 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/cti_result \
                                                                --data $cti_scp --use_gpu --gpu $gpus || wechat echo "cti torchrun error"
                                mkdir -p $result_path/$model_id/cti_result
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/cti_result/embeddings --test_data $result_path/$model_id/cti_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/cti_result/scores --trials $trials_cti || wechat echo "cti compute_score_metrics error"
                fi

                if [ "$trial_class" == '3dspeaker' ]; then
                        echo "3dspeaker dataset"
                        if [ ! -d $result_path/$model_id/3dspeaker_result/embeddings ]; then
                                torchrun --nproc_per_node=$nj --master_port=55688 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/3dspeaker_result \
                                                                --data $speaker_scp --use_gpu --gpu $gpus || wechat echo "3d torchrun error"
                                mkdir -p $result_path/$model_id/3dspeaker_result
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/3dspeaker_result/embeddings --test_data $result_path/$model_id/3dspeaker_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/3dspeaker_result/scores --trials $trials_3dspeaker || wechat echo "3d compute_score_metrics error"
                fi

                if [ "$trial_class" == 'male' ]; then
                        echo "male dataset"
                        if [ ! -d $result_path/$model_id/male/embeddings ]; then
                                torchrun --nproc_per_node=$nj --master_port=54688 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/male_result \
                                                                --data $male_scp --use_gpu --gpu $gpus || wechat echo "3d torchrun error"
                                mkdir -p $result_path/$model_id/male_result
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/male_result/embeddings --test_data $result_path/$model_id/male_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/male_result/scores --trials $trials_male || wechat echo "3d compute_score_metrics error"
                fi  

                if [ "$trial_class" == 'female' ]; then
                        echo "female dataset"
                        if [ ! -d $result_path/$model_id/female/embeddings ]; then
                                torchrun --nproc_per_node=$nj --master_port=53688 speakerlabduanyibo/bin/extract.py --exp_dir $result_path/$model_id/female_result \
                                                                --data $female_scp --use_gpu --gpu $gpus || wechat echo "3d torchrun error"
                                mkdir -p $result_path/$model_id/female_result
                        fi
                        python speakerlabduanyibo/bin/compute_score_metrics.py --enrol_data $result_path/$model_id/female_result/embeddings --test_data $result_path/$model_id/female_result/embeddings \
                                                                        --scores_dir $result_path/$model_id/female_result/scores --trials $trials_female || wechat echo "female compute_score_metrics error"
                fi 
        done
        wechat echo "$model_id done"
done
                