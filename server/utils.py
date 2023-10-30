import subprocess
from loguru import logger
import os

import numpy as np
import cfg
import redis
import struct


def run_cmd(cmd, check=True, util_exist=None):
    """run shell command.
    Args:
        cmd (string): shell command.
    Returns:
        string: result of shell command.
    """
    # logger.info(f"Run command: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if util_exist:
        test_time = 0
        # TODO:
        if ((not os.path.exists(util_exist)) or os.path.getsize(util_exist) < 1000) and test_time < 10:
            test_time = test_time + 1
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check:
        if result.returncode != 0:
            logger.info(f"Command: {cmd}, result output: {result.stdout}")
            logger.error("run command error!")
            raise Exception(f"run command error! \n cmd :{cmd} cmd result: {result.stdout}")
        else:
            logger.info(f"Command: {cmd}, result output: {result.stdout}")
    else:
        logger.info(f"Command: {cmd}")
    return result.stdout.decode('utf-8')


def remove_father_path(filepath):
    father_path = os.path.dirname(filepath)
    run_cmd(f"rm -rf {father_path}")


def save_file(file, spkid, channel=0, start=0, length=999, sr=16000):
    """
    save file to local
    """
    end = start + length
    receive_path = cfg.RECEIVE_PATH
    spk_dir = os.path.join(receive_path, str(spkid))
    os.makedirs(spk_dir, exist_ok=True)
    filename = file.filename.split("/")[-1]
    raw_save_path = os.path.join(spk_dir, f"raw_{filename}")
    convert_save_path = os.path.join(spk_dir, f"conver_{filename}")
    logger.info(f"Save raw file to {raw_save_path}, conver file to {convert_save_path}")
    file.save(raw_save_path)
    # conver to wav
    # TODO:
    cmd = f"ffmpeg -i {raw_save_path} -y  -ss {start} -to {end} -ar {sr}  -ac 1 -vn -map_channel 0.0.{channel} -y  {convert_save_path}"
    run_cmd(cmd, util_exist=convert_save_path)
    return convert_save_path


def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    shape = struct.pack(">II", 256, 1)
    encoded = shape + a.tobytes()
    r.set(n, encoded)
    return


def to_database(embedding_npy, spkid, use_model_type, mode="register"):
    embedding_npy = np.array(embedding_npy, dtype=np.float32)
    if not use_model_type:
        logger.error(f"No use_model_type. spkid:{spkid}")
        return False

    if mode == "register":
        db = cfg.REDIS["register_db"]
    else:
        db = cfg.REDIS["test_db"]
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=db,
        password=cfg.REDIS["password"],
    )
    toRedis(r, embedding_npy, f"{use_model_type}_{spkid}")

    return True


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


def get_embeddings_for_spkid(spkid):
    """
    Get embeddings for spkid
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    embeddings = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        uuid_db = key.split("_")[-1]
        if uuid_db == spkid:
            embedding = fromRedis(r, key)
            embeddings[key.replace("_"+uuid_db, '')] = embedding

    return embeddings


def get_spkid():
    """
    Get spkid
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    spkids = set()
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        uuid_db = key.split("_")[-1]
        spkids.add(uuid_db)

    return spkids


def hello():
    with open("hello.txt", "w") as f:
        f.write("hello world")
    print("hello world")
