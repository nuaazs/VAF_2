import yaml
import io
import time
from cryptography.fernet import Fernet

key = b'vlu3T4bs2WWK5lc2QB-yKvGc_20P4gh6TqD7nhuh7pU='
year_month = time.strftime("%Y-%m", time.localtime())
key += year_month.encode()


cipher_suite = Fernet(key)

# 读取 YAML 文件
with open('/home/xuekaixiang/workplace/cmb_test/server/dguard/files/yaml/encrypted293_cjsd.yaml', 'r') as file:
    data = file.read()

# 加密数据
encrypted_data = cipher_suite.encrypt(data.encode())

# 写入加密后的数据到一个新文件
with open('encrypted293_cjsd.yaml', 'wb') as file:
    file.write(encrypted_data)

# # 解密已加密的文件
# with open('encrypted_config.yaml', 'rb') as file:
#     encrypted_data = file.read()

# decrypted_data = cipher_suite.decrypt(encrypted_data)

# # 将解密的数据写入新文件
# with open('decrypted_config.yaml', 'w') as file:
#     file.write(decrypted_data.decode())


def yaml_decryption(encryt_file):
    with open(encryt_file, 'rb') as fr:
        encrypted_data = fr.read()
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    model = yaml.load(b)
    return model

print("加密和解密完成")
