import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 生成随机数据 (1000条数据，每个数据192维度)
data = np.random.rand(10, 192)

# 生成给定数据 (192维度)
given_data = np.random.rand(1, 192)

# # 计算所有数据与给定数据的余弦相似度
# cosine_similarities = cosine_similarity(data, given_data)
# print(f"cosine_similarities: {cosine_similarities}")
# # 提取TOP1相似度最高的索引
# top_indices = np.argsort(cosine_similarities.ravel())[-1]
# print(f"top_indices: {top_indices}")
# # 打印TOP10相似度最高的数据
# print(f"TOP 1: Similarity = {cosine_similarities[top_indices][0]}")



if os.path.exists("/tmp/sss"):
    shutil.rmtree("/tmp/sss")
    print("done")