
import numpy as np
import sys
sys.path.append("../")
import cfg

with open("./black_id_all.txt", "r") as f:
    black_id_all = f.readlines()
    black_id_all = [i.strip() for i in black_id_all]

read_data = np.fromfile("./resnet293cjsdlm_vectorDB.bin", dtype=np.float32)
read_data = read_data.reshape(-1, 256)
read_data = read_data.tolist()
print(len(black_id_all))
print(len(read_data))

print(black_id_all[100])
print(read_data[100])
