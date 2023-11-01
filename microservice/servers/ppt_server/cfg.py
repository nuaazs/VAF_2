
HOST = "http://192.168.3.169"
VAD_URL = f"{HOST}:8899/energy_vad/file"  # VAD
LANG_URL = f"{HOST}:5002/lang_classify"  # 语种识别
ENCODE_URL = f"{HOST}:8899/encode/file"  # 提取特征
CLUSTER_URL = f"{HOST}:5011/cluster"  # cluster
ASR_URL = f"{HOST}:5000/transcribe/file"  # ASR

ENCODE_MODEL_NAME = "resnet101_cjsd"    # 使用的模型类型
ENCODE_MODEL_LIST = ["resnet101_cjsd", "resnet221_cjsd_lm", "resnet293_cjsd_lm"]  # 注册模型列表
BLACK_TH = {"resnet101_cjsd": 0.78}

CLUSTER_MIN_SCORE_THRESHOLD = 0.8  # cluster的阈值
COMPARE_SCORE_THRESHOLD = 0.6  # 声纹比对的阈值
VAD_MIN_LENGTH = 10  # 有效音频最小长度

TMP_FOLDER = '/tmp/ppt'
SHOW_PUBLIC = True  # 是否是对外展示的版本
PUBLIC_HOST = 'http://81.69.253.47'  # 公网映射地址

MYSQL = {
    "host": "192.168.3.169",
    "port": 3306,
    "db": "si",
    "username": "zhaosheng",
    "passwd": "Nt3380518",
    "black_table_name": "black_speaker_info_ppt",
    "hit_table_name": "hit_ppt",
}

REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 10,
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
    "register_bucket_name": "black-ppt",
    "test_bucket_name": "test",
}
