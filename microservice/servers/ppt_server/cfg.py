
HOST = "http://192.168.3.169"
VAD_URL = f"{HOST}:5005/energy_vad/file"  # VAD
LANG_URL = f"{HOST}:5002/lang_classify"  # 语种识别
ENCODE_URL = f"{HOST}:7701/encode"  # 提取特征
CLUSTER_URL = f"{HOST}:5011/cluster"  # cluster
ASR_URL = f"{HOST}:5000/transcribe/file"  # ASR

ENCODE_MODEL_NAME = "eres2net"    # 使用的模型类型
ENCODE_MODEL_NAME_LIST = ["eres2net"]
BUCKET_NAME = "black-ppt"  # 采集的bucket
COMPARE_BUCKET_NAME = "test"  # 比对的bucket
CLUSTER_MIN_SCORE_THRESHOLD = 0.8  # cluster的阈值
COMPARE_SCORE_THRESHOLD = 0.6  # 声纹比对的阈值
MIN_LENGTH = 10

TEMP_PATH = '/tmp'
SHOW_PUBLIC = True  # 是否是对外展示的版本
PUBLIC_IP = 'http://106.14.148.126'  # 公网映射地址

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
