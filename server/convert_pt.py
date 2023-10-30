import io
import time
import torch
from cryptography.fernet import Fernet

#指定初始字符串
key = b'vlu3T4bs2WWK5lc2QB-yKvGc_20P4gh6TqD7nhuh7pU='
year_month = time.strftime("%Y-%m", time.localtime())
#添加月份，作用是设置有效期，次月失效
key += year_month.encode()


def model_encryption(pth_file, encryp_file):
    """
    加密模型
    :param pth_file: 模型路径
    :param encryp_file: 加密后模型路径
    """
    model = torch.load(pth_file)
    b = io.BytesIO()
    torch.save(model, b)
    b.seek(0)
    pth_bytes = b.read()
    encrypted_data = Fernet(key).encrypt(pth_bytes)
    with open(encryp_file, 'wb') as fw:
        fw.write(encrypted_data)


def model_decryption(encryt_file):
    """
    解密模型
    :param encryt_file: 加密模型路径
    :return: model
    """
    with open(encryt_file, 'rb') as fr:
        encrypted_data = fr.read()
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    model = torch.load(b)
    return model


pth_file = '/home/xuekaixiang/workplace/cmb_test/server/dguard/files/pt/resnet293_cjsd_lm.pt'
model_encryption(pth_file, 'encrypted293_cjsd_lm.pt')
