RECEIVE_PATH = '/tmp/dingxiaoyu/'
DEVICE = 'cuda:0'
ENCODE_MODEL_LIST = ["encrypted101_cjsd", "encrypted221_cjsd_lm", "encrypted293_cjsd_lm"]

VAD_LENGTH = 3  # 注册最小整数时长（秒）

REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 10,
    "test_db": 2,
    "password": "",
}
