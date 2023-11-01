WORKERS = 4
SR = 16000
LANGUAGE_DEVICE = "cuda:0"
TEMP_PATH = "/tmp"

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
