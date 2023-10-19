###########################################
# 公司内部服务配置文件
###########################################
HOST = "http://192.168.3.199"
LANG_URL = f"{HOST}:5002/lang_classify"  # 语种识别
CLUSTER_URL = f"{HOST}:5011/cluster"  # cluster
ASR_URL = f"{HOST}:5000/transcribe/file"  # ASR

# 注册模型列表
ENCODE_MODEL_LIST = ["resnet101_cjsd", "resnet221_cjsd_lm"]

TMP_FOLDER = "/tmp/dingxiaoyu"

DEVICE = 'cuda:1'

# VAD长度
VAD_MIN_LENGTH = 10

# 命中阈值
HIT_SCORE_THRESHOLD = 0.8

# 去重指定使用的模型及阈值
BLACK_TH = {"resnet101_cjsd": 0.78}


MYSQL = {
    "host": "127.0.0.1",
    "port": 3306,
    "db": "si",
    "username": "root",
    "passwd": "longyuan",
    "black_table_name": "black_speaker_info",
    "hit_table_name": "cjsd_hit_speaker_info",
}

REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 0,  # 2.7W
    "test_db": 2,
    "password": "",
}

MINIO = {
    "host": "192.168.3.199",
    "port": 9000,
    "access_key": "longyuan",
    "secret_key": "longyuan",
    "test_save_days": 30,
    "register_save_days": -1,
    "register_bucket_name": "black",
    "hit_bucket_name": "hit",
}
