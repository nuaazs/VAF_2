WORKERS = 4
SR = 16000
WAV_START = 0
WAV_LENGTH = 999
TIME_TH = 2
DEVICE = "cuda:0"
THREADS = 10
TEST_THREADS = 20
MIN_LENGTH_REGISTER = 10
WORKER_CONNECTIONS = 1000
PORT = 7777
# VAD_BIN="/home/zhaosheng/speaker-diarization/bin/vad"
MAX_WAV_NUMBER = 10
MODEL_PATH = {"ECAPATDNN": "/VAF/src/nn/ECAPATDNN", "CAMPP": "/VAF/src/nn/CAMPP"}
# damo="/home/zhaosheng/asr_damo_websocket/online/damo"
damo = "./models"


TEMP_PATH = "/tmp"
SR = 16000

NNVAD_DEVICE = "cuda:0"
CRDNN_DEVICE = "cuda:0"
ENCODE_CAMPP_DEVICE = "cuda:0"
ENCODE_ECAPATDNN_DEVICE = "cuda:0"
ENCODE_ERES2NET_DEVICE = "cuda:0"
GENDER_DEVICE = "cuda:0"
LANGUAGE_DEVICE = "cuda:0"
PUNC_PYTHON_DEVICE = "cuda:0"
ASR_PYTHON_DEVICE = "cuda:0" 


"""
[x] "CAMPP_200k"
[-] "CAMPP_3dspeaker"
[x] "CAMPP_3dspeakerphone"
[ ] "CAMPP_200k_Fvoxphone"
[ ] "CAMPP_200k_F3dspeakerphone"
[x] "CAMPP_voxphone"
[x] "ECAPATDNN_voxphone"
[ ] "ECAPATDNN_3dspeaker"
[x] "ERES2NETLARGE_200k"
[ ] "ERES2NETLARGE_3dspeakerphone"
[ ] "ERES2NETLARGE_3dspeaker"
[ ] "ERES2NETLARGE_200k_Fvoxphone"
[ ] "ERES2NETLARGE_200k_F3dspeakerphone"
"""

# ENCODE_MODEL_LIST = ["CAMPP_200k","CAMPP_200k_Fvoxphone","CAMPP_200k_F3dspeakerphone","CAMPP_voxphone","ECAPATDNN", "CAMPP_voxphone","ERES2NET_Large"] #["CAMPP_200k","CAMPP_200k_Fvoxphone","CAMPP_200k_F3dspeaker","ECAPATDNN", "CAMPP_voxphone","ERES2NET_Large"] # ["ERES2NET_Large","ERES2NET_Base","ECAPATDNN", "CAMPP_voxphone", "CAMPP_200k","CAMPP_200k_Fvoxphone","CAMPP_200k_F3dspeaker.py"]
ENCODE_MODEL_LIST = ["eres2net"]
BLACK_TH = {"ECAPATDNN": 0.78, "CAMPP": 0.78, "eres2net": 0.78}
EMBEDDING_LEN = {"ECAPATDNN": 192, "CAMPP": 512, "ERES2NET": 192, "eres2net": 192}

MYSQL = {
    "host": "192.168.3.169",
    "port": 3306,
    "db": "si",
    "username": "zhaosheng",
    "passwd": "Nt3380518",
}

REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 0,
    "test_db": 2,
    "password": "",
}

MINIO = {
    "host": "192.168.3.169",
    "port": 9000,
    "access_key": "zhaosheng",
    "secret_key": "zhaosheng",
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
    "zs": b"zhaoshengzhaoshengnuaazs",
}

###########################################
# 以下为speaker_diarization_server的配置
HOST = "http://192.168.3.169"
VAD_URL = f"{HOST}:5005/energy_vad/file"  # VAD
LANG_URL = f"{HOST}:5002/lang_classify"  # 语种识别
ENCODE_URL = f"{HOST}:5001/encode"  # 提取特征
CLUSTER_URL = f"{HOST}:5011/cluster"  # cluster
ASR_URL = f"{HOST}:5000/transcribe/file"  # ASR

USE_MODEL_TYPE = "eres2net"    # 使用的模型类型
BUCKET_NAME = "check-for-speaker-diraization"  # 存储的bucket
CLUSTER_MIN_SCORE_THRESHOLD = 0.5  # cluster的阈值
