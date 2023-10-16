#!/bin/bash


# ==================================================================
# 脚本作用描述
# ==================================================================
# 原始数据路径
# DATA_ROOT=/mnt/data/cmb2_vad
# DATA_SET_NAME=cmb2
# DIST_ENROLL_ROOT=""
# DIST_TEST_ROOT=""
# DIST_DISTURB_ROOT=""
# random_seed=123
# SET RANDOM SEED
# ${DATA_ROOT}/<spkid>/<filename>.wav

# TEST_SPEAKER_NUM=100
# DISTURB_SPEAKER_NUM= ALL - TEST_SPEAKER_NUM

# STEP1 : 随机选择TEST_SPEAKER_NUM个说话人数据，做如下操作
# 1. 将所有filename="enroll"的文件复制到 ${DIST_ENROLL_ROOT}/<spkid>/<filename>.wav
# 2. 将所有filename!="enroll"的文件复制到 ${DIST_TEST_ROOT}/<spkid>/<filename>.wav
# 3. 生成${DIST_ENROLL_ROOT}/${DATA_SET_NAME}_enroll.txt
# ---其中内容为<spkid> ${DIST_ENROLL_ROOT}/<spkid>/<filename>.wav
# 4. 生成${DIST_TEST_ROOT}/${DATA_SET_NAME}_test.txt
# ---其中内容为<spkid> ${DIST_TEST_ROOT}/<spkid>/<filename>.wav

# STEP2 : 选择剩下的DISTURB_SPEAKER_NUM个说话人数据，做如下操作
# 1. 将所有wav文件复制到 ${DIST_DISTURB_ROOT}/<spkid>/<filename>.wav
# 3. 生成${DIST_DISTURB_ROOT}/${DATA_SET_NAME}_disturb.txt
# ---其中内容为<spkid> ${DIST_DISTURB_ROOT}/<spkid>/<filename>.wav





# ==================================================================
# 可配置参数
# ==================================================================
DATA_ROOT=/mnt/data/cmb2_vad
DATA_SET_NAME=cmb2
random_seed=123
TEST_SPEAKER_NUM=100




# ==================================================================
# 脚本内容
# ==================================================================

DIST_ENROLL_ROOT="./data/${DATA_SET_NAME}_enroll"
DIST_TEST_ROOT="./data/${DATA_SET_NAME}_test"
DIST_DISTURB_ROOT="./data/${DATA_SET_NAME}_disturb"
# 设置随机种子
RANDOM=${random_seed}

# 清空输出目录
# rm -rf "${DIST_ENROLL_ROOT}"
# rm -rf "${DIST_TEST_ROOT}"
# rm -rf "${DIST_DISTURB_ROOT}"
mkdir -p "${DIST_ENROLL_ROOT}"
mkdir -p "${DIST_TEST_ROOT}"
mkdir -p "${DIST_DISTURB_ROOT}"

# STEP1: 随机选择 TEST_SPEAKER_NUM 个说话人数据
# 获取所有的说话人列表
speaker_list=($(find "${DATA_ROOT}" -type d -name '[0-9]*' | shuf))

# 获取需要作为测试集的说话人列表
test_speaker_list=("${speaker_list[@]:0:${TEST_SPEAKER_NUM}}")

# 获取剩余的说话人列表作为干扰集
disturb_speaker_list=("${speaker_list[@]:${TEST_SPEAKER_NUM}}")

# 复制 enroll 和 test 文件，并生成文件列表
for speaker in "${test_speaker_list[@]}"; do
    spkid=$(basename "${speaker}")
    find "${speaker}" -type f -name "enroll.wav" -exec cp "{}" "${DIST_ENROLL_ROOT}/${spkid}/" \;
    find "${speaker}" -type f -not -name "enroll.wav" -exec cp "{}" "${DIST_TEST_ROOT}/${spkid}/" \;
    find "${DIST_ENROLL_ROOT}/${spkid}" -type f -name "*.wav" -exec echo "${spkid} {}" \; >> "${DIST_ENROLL_ROOT}/${DATA_SET_NAME}_enroll.txt"
    find "${DIST_TEST_ROOT}/${spkid}" -type f -name "*.wav" -exec echo "${spkid} {}" \; >> "${DIST_TEST_ROOT}/${DATA_SET_NAME}_test.txt"
done

# STEP2: 选择剩下的 DISTURB_SPEAKER_NUM 个说话人数据
for speaker in "${disturb_speaker_list[@]}"; do
    spkid=$(basename "${speaker}")
    find "${speaker}" -type f -exec cp "{}" "${DIST_DISTURB_ROOT}/${spkid}/" \;
    find "${DIST_DISTURB_ROOT}/${spkid}" -type f -name "*.wav" -exec echo "${spkid} {}" \; >> "${DIST_DISTURB_ROOT}/${DATA_SET_NAME}_disturb.txt"
done



# 脚本结束