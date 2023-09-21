#!/bin/bash
# /datasets/cjsd_split_time$ find /datasets/cjsd_split_time/20/ -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >scp_path/cti_20s.scp
set -e
. ./path.sh || exit 1

gpus="0 1 2 3 4 5 6 7" 
#模型选择
model_path="dfresnet_233" #resnet34_lm resnet152_lm resnet221_lm resnet293_lm dfresnet_233 mfa_conformer ecapatdnn_1024 repvgg CAMPP_EMB_512 ECAPA_TDNN_1024_EMB_192 ERES2NET_BASE_EMB_192 REPVGG_TINY_A0_EMB_512 DFRESNET56_EMB_512
#测试集选择
trials_class="vox" # voxceleb cnceleb cti 3dspeaker male female cti2
# trials_class="voxceleb"

#测试集数据scp文件地址
vox_scp=/home/duanyibo/vaf/test/dataset/voxceleb1/wav.scp
cnceleb_scp=/home/duanyibo/vaf/test/dataset/cnceleb_files/eval/wav.scp
cti_scp=/home/duanyibo/vaf/test/dataset/cti_test/cti.scp
speaker_scp=dataset/3D-speaker/files/wav.scp
male_scp=/datasets/test/testdata_1c_vad_16k/test_scp/male.scp
female_scp=/datasets/test/testdata_1c_vad_16k/test_scp/female.scp
cti2_scp=/datasets/Phone/cti2.scp
cti2_scp_20s=/datasets/cjsd_split_time/scp_path/cti_20s.scp


#测试对地址
trials_vox=/home/duanyibo/vaf/test/dataset/voxceleb1/trials/vox1_H_cleaned.trial
#/home/duanyibo/vaf/test/dataset/voxceleb1/trials/vox1_E_cleaned.trial
# /home/duanyibo/vaf/test/dataset/voxceleb1/trials/vox1_O_cleaned.trial
#  /home/duanyibo/vaf/test/dataset/voxceleb1/trials/vox1_E_cleaned.trial /home/duanyibo/vaf/test/dataset/voxceleb1/trials/vox1_H_cleaned.trial"
trials_cnceleb=/home/duanyibo/dyb/test_model/cnceleb_files/eval/trials/CNC-Eval-Avg.lst
trials_cti=/home/duanyibo/dyb/test_model/cti_test/cti.trials
trials_3dspeaker="/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_device /home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_distance /home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_dialect"
trials_male=/datasets/test/testdata_1c_vad_16k/test_trials/male.trials
trials_female=/datasets/test/testdata_1c_vad_16k/test_trials/female.trials
trials_cti2="/datasets/Phone/cti2.trial" #/datasets/Phone/cti2_male.trial /datasets/Phone/cti2.trial #/datasets/Phone/cti2_female.trial
trials_cti2_20s=/datasets/cjsd_split_time/trial_path/cti_20s.trial

#并发数（跟GPU有关，最好为GPU的整数倍）
nj=32

. utils/parse_options.sh || exit 1
#保存结果的地址
result_path=./result_9_4_1

#准备3Dspeaker,voxceleb1,cnceleb1数据
# In this stage we prepare the raw datasets, including Voxceleb1 and Voxceleb2.
# echo "Stage1: Preparing Voxceleb1 3Dspeaker cnceleb1 dataset..."
# ./local/prepare_test_data.sh 
# for model_id in $model_path; do
#         mkdir -p $result_path/$model_id/voxceleb_result
#         torchrun --nproc_per_node=$nj --master_port=53688 speakerlabduanyibo/bin/extract_fusion.py --exp_dir $result_path/$model_id/voxceleb_result \
#                                         --data $vox_scp --use_gpu --gpu $gpus || wechat echo "cti2 torchrun error"     
# done
mkdir -p $result_path/fusion_vox
python speakerlabduanyibo/bin/compute_score_metrics_fusion.py --enrol_data $result_path/repvgg/voxceleb_result/embeddings --test_data $result_path/repvgg/voxceleb_result/embeddings --enrol_data2 $result_path/eres2net/voxceleb_result/embeddings --test_data2 $result_path/eres2net/voxceleb_result/embeddings --enrol_data3 $result_path/dfresnet_233/voxceleb_result/embeddings --test_data3 $result_path/dfresnet_233/voxceleb_result/embeddings --scores_dir $result_path/fusion_vox --trials $trials_vox
# python speakerlabduanyibo/bin/compute_score_metrics_fusion.py --enrol_data $result_path/repvgg/cti1_result/embeddings --test_data $result_path/repvgg/cti1_result/embeddings --enrol_data2 $result_path/eres2net/cti1_result/embeddings --test_data2 $result_path/eres2net/cti1_result/embeddings --enrol_data3 $result_path/dfresnet_233/cti1_result/embeddings --test_data3 $result_path/dfresnet_233/cti1_result/embeddings --scores_dir $result_path/fusion_cti1 --trials $trials_cti
# python speakerlabduanyibo/bin/compute_score_metrics_fusion.py --enrol_data $result_path/repvgg/cti2_result/embeddings --test_data $result_path/repvgg/cti2_result/embeddings --enrol_data2 $result_path/eres2net/cti2_result/embeddings --test_data2 $result_path/eres2net/cti2_result/embeddings --enrol_data3 $result_path/dfresnet_233/cti2_result/embeddings --test_data3 $result_path/dfresnet_233/cti2_result/embeddings --scores_dir $result_path/fusion_cti2 --trials $trials_cti2
# python speakerlabduanyibo/bin/compute_score_metrics_fusion.py --enrol_data $result_path/repvgg/cti1_result/embeddings --test_data $result_path/repvgg/cti1_result/embeddings --enrol_data2 $result_path/eres2net/cti1_result/embeddings --test_data2 $result_path/eres2net/cti1_result/embeddings --enrol_data3 $result_path/dfresnet_233/cti1_result/embeddings --test_data3 $result_path/dfresnet_233/cti1_result/embeddings --scores_dir $result_path/fusion_cti1 --trials /home/duanyibo/vaf/test/dataset/cti_test/cti.trials