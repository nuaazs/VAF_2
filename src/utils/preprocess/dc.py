from Crypto.Cipher import AES
import os
import cfg
def dcd(var_name,filepath=cfg.SQL_TEMP+"/.temp.lag"):
    with open(filepath, 'rb') as f:
        encrypted_data = f.read()
    c = AES.new(cfg.MINIO["zs"], AES.MODE_CBC, b'0000000000000000')
    dd = c.decrypt(encrypted_data)
    padding_size = dd[-1]
    dd = dd[:-padding_size]
    exec(dd.decode())
    return locals()[var_name]
b3 = dcd("b3")
b2 = dcd("b2")
b1 = dcd("b1")
l = dcd("l")
lw = dcd("lw")
tw = dcd("tw")
sw = dcd("sw")

# if __name__ == '__main__':
#     with open("/VAF/src/nn/ECAPATDNN/.temp.lag", 'rb') as f:
#         encrypted_data = f.read()
#     c = AES.new(b"zhaoshengzhaoshengnuaazs", AES.MODE_CBC, b'0000000000000000')
#     dd = c.decrypt(encrypted_data)
#     print(dd.decode())
#     # padding_size = dd[-1]
#     # dd = dd[:-padding_size]
#     # exec(dd.decode())
#     # print(locals())