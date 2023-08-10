import pickle
import struct
import redis
import numpy as np
import cfg

# log
from loguru import logger


def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    shape = struct.pack(">II", 192, 1)
    encoded = shape + a.tobytes()
    r.set(n, encoded)
    return


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


def deletRedis(r, n):
    if r.keys(f"*{n}*"):
        r.delete(*r.keys(f"*{n}*"))
    return


def get_embedding(spkid):
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )

    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        now_id = key.split("_")[1]
        if now_id == spkid:
            embedding = fromRedis(r, key)

    return embedding


def get_embeddings(use_model_type="ERES2NET_Base"):
    """
    获取黑库中所有指定模型的emb
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    all_embedding = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        embedding_type = str(key.split("_")[0])
        if embedding_type != use_model_type:
            continue
        spkid = key.split("_")[1]
        embedding_1 = fromRedis(r, key)
        all_embedding[spkid] = embedding_1
    logger.info(f"Total : {len(all_embedding.keys())} embeddings in database.Use model type:{use_model_type}")
    return all_embedding


def to_database(embedding, spkid, use_model_type, mode="register"):
    if not use_model_type:
        logger.error(f"No use_model_type. spkid:{spkid}")
        return False
    embedding_npy = embedding.detach().cpu().numpy()

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


def delete_by_key(blackbase, spkid):
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    deletRedis(r, spkid)
    return


def save_redis_to_pkl():
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    all_embedding = {}
    for key in r.keys():
        key = key.decode("utf-8")
        spkid = key
        embedding_1 = fromRedis(r, key)
        all_embedding[spkid] = {"embedding_1": embedding_1}
    with open(cfg.BLACK_BASE, "wb") as f:
        pickle.dump(all_embedding, f, pickle.HIGHEST_PROTOCOL)


