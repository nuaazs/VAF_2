# coding = utf-8
# @Time    : 2022-09-05  09:43:48
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: vaf service configuration file.

#######################################################
############## 1. System Configuration ################
#######################################################

WORKERS = 1

SR = 8000

DEVICE = "cuda:0"

# DEVICE = "cpu"

THREADS = 20
TEST_THREADS = 20
WORKER_CONNECTIONS = 1000

PORT = 8191

#######################################################
####################### 2. VAD ########################
#######################################################

large_chunk_size = 30
small_chunk_size = 10
overlap_small_chunk = True
apply_energy_VAD = True
double_check = True
close_th = 0.1
len_th = 0.3
activation_th = 0.5
deactivation_th = 0.25
en_activation_th = 0.6
en_deactivation_th = 0.6
speech_th = 0.5

VAD_MODEL = "vad_8k_en_phone_crdnns"

#######################################################
################### 3. Pre-process ####################
#######################################################

VAD_TYPE = "speechbrain"

WAV_START = 0

CLASSIFY = False

CHECK_DUPLICATE = False 
DUPLICATE_TYPE = "remove_old"

LOG_PHONE_INFO = False

#######################################################
################### 4. Thresholds #####################
#######################################################

BLACK_TH = 0.78

MIN_LENGTH_REGISTER = 1

MIN_LENGTH_TEST = 1

WAV_LENGTH = 90

WAV_CHANNEL = 0

ONLY_VAD = 0

# 是否返回VAD处理后tensor
SHOW_VAD_LIST = 0

USE_FBANK = 0

SAVE_PREPROCESSED_OSS = False

#######################################################
#################### 5. Databases #####################
#######################################################

MYSQL = {
    "host": "192.168.3.201",
    "port": 3306,
    "db": "si",
    "username": "root",
    "passwd": "longyuan",
}

REDIS = {
    "host": "192.168.3.202",
    "port": 6379,
    "register_db": 1,
    "test_db": 2,
    "password": "",
}

MINIO = {
    "host": "192.168.3.202",
    "port": 9000,
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "test_save_days": 30,
    "register_save_days": -1,
    "register_raw_bucket": "register_raw",
    "register_preprocess_bucket": "register_preprocess",
    "test_raw_bucket": "test_raw",
    "test_preprocess_bucket": "test_preprocess",
    "pool_raw_bucket": "pool_raw",
    "pool_preprocess_bucket": "pool_preprocess",
    "black_raw_bucket": "black_raw",
    "black_preprocess_bucket": "black_preprocess",
    "white_raw_bucket": "white_raw",
    "white_preprocess_bucket": "white_preprocess",
}
BUCKETS = ["raw", "preprocess", "preprocessed", "testing", "sep"]

SQL_TABLES = ["si", "si_pool"]
SQL_FILES = {
    "si": "./database/si_0420.sql",
    "si_pool": "./database/si_0420.sql",
}

SERVER_INFO = {"name": "lyxx-192.168.3.202", "ip": "192.168.3.202", "port": PORT}

# =====================update=============================
ASR_SERVER = "http://192.168.3.202:8000/asr"
BLACK_WORDS_PATH = "./black_words.txt"
TEST_WAVS_DIR = "./test_wavs"
ECAPA_SR = 16000
TEMP_PATH = "/tmp"
SAVE_FBANK = True
apply_energy_VAD_before = False
# =====================update=============================
filter_mandarin = False
filter_mandarin_threshold = 0.9


# =====================update=============================
DOUBLE_MODEL_CHECK = True
DOUBLE_MODEL_CHECK_THRESHOLD = 0.5
DOUBLE_MODEL_CHECK_MODEL = "CAMpp-16k-phone_1"
USE_CAMPP = False