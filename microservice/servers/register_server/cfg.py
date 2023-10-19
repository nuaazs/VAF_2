###########################################
# 此处为register server的配置文件
# this is the config file for register server

HOST = "http://192.168.3.169"
VAD_URL = f"{HOST}:5005/energy_vad/file"  # VAD
LANG_URL = f"{HOST}:5002/lang_classify"  # 语种识别
ENCODE_URL = f"{HOST}:5001/encode"  # 提取特征
CLUSTER_URL = f"{HOST}:5011/cluster"  # cluster
ASR_URL = f"{HOST}:5000/transcribe/file"  # ASR

ENCODE_MODEL_LIST = ["resnet101_cjsd", "resnet221_cjsd_lm", "resnet293_cjsd_lm"]  # 注册模型列表
BLACK_TH = {"resnet101_cjsd": 0.78}  # 去重指定使用的模型及阈值
TMP_FOLDER = "/tmp/dingxiaoyu"
DEVICE = 'cuda:0'
VAD_MIN_LENGTH = 10  # VAD长度


MYSQL = {
    "host": "192.168.3.169",
    "port": 3306,
    "db": "si",
    "username": "zhaosheng",
    "passwd": "Nt3380518",
    "black_table_name": "black_speaker_info",
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
    "register_bucket_name": "black",
}
