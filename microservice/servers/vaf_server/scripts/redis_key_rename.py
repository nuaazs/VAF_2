import redis

# 连接到 Redis 服务器
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 获取所有符合条件的键
keys_to_rename = redis_client.keys('resnet*')
print(len(keys_to_rename))

# 遍历并重命名键
for old_key in keys_to_rename:
    old_key = old_key.decode('utf-8')
    spkid = old_key.split('_')[-1]
    new_key = old_key.replace('_'+spkid, '').replace('_', '')+"_"+spkid
    redis_client.rename(old_key, new_key)
    print(old_key, new_key)

# 关闭 Redis 连接
redis_client.close()
