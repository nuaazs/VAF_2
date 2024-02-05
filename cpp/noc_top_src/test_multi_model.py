# import ctypes
import numpy as np
import torch

cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# from ctypes import c_float, c_int, POINTER, c_char_p

# 生成 vectorA
# vectorA = np.random.rand(1, 256).flatten().astype(np.float32)
# # 将 vectorA 保存到 test.txt ，一行一个数
# with open('test.txt', 'w') as f:
#     for item in vectorA:
#         f.write(str(item) + '\n')

# 读取 test.txt 中的数据,转为 float32 npy
vectorA = np.loadtxt('test.txt', dtype=np.float32)
print(vectorA.shape)



def get_vectorDB(bins=["/VAF/cpp/noc_top_src/test_data/101.bin","/VAF/cpp/noc_top_src/test_data/221.bin","/VAF/cpp/noc_top_src/test_data/293.bin"]):
    dbs = []
    for bin in bins:
        db = np.fromfile(bin, dtype=np.float32)
        db = db.reshape(-1, 256)
        dbs.append(db)
    return dbs


def retrieve_python(A):
    A = np.array(A)
    A = A.reshape(-1, 256)
    db_list = get_vectorDB()
    score_list = []
    result = []
    for db in db_list:
        for _index,B in enumerate(db):
            B = B.reshape(-1, 256)
            score = cosine_similarity(torch.from_numpy(A), torch.from_numpy(B))
            if len(result) <= _index:
                result.append([score/len(db_list),_index])
            else:
                result[_index][0] += score/len(db_list)
    # sort result,by score
    result.sort(key=lambda x:x[0],reverse=True)
    return result[:12]

r = retrieve_python(vectorA)
print(r)
