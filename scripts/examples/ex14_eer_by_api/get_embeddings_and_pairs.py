import requests
from tqdm import tqdm
import numpy as np

import requests
from tqdm import tqdm
import numpy as np
import cfg
import os
def get_embedding(file_path,model="ERES2NET"):
    url = "http://192.168.3.169:5001/encode"
    payload={
        "spkid": "123456",
        "channel": 0,
        "filelist": f"local://{file_path}",
        "save_oss": "False",
        "score_threshold": 0.7
    }
    headers = {
    'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.json())
    return_data = {}
    # for _model_name in model.split(","):
    embeddings = response.json()["file_emb"][model]["embedding"]
    temp_file = [_key for _key in embeddings.keys()][0]
    emb_result = embeddings[temp_file]
    return_data = np.array(emb_result)
        # print(len(emb_result))
    # print(return_data.shape)
    return return_data



if __name__ == "__main__":
    embeddings = {}
    all_wavs = []
    # if npy exist, just load
    if os.path.exists(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy") and cfg.LOAD_NPY:
        embeddings = np.load(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",allow_pickle=True).item()
        for phone in embeddings:
            phone_file_nums = 0
            for filename in embeddings[phone]:
                all_wavs.append(os.path.join(cfg.DATA_FOLDER,phone,filename))
                phone_file_nums += 1
        print("Load npy file successfully.")
    else:
        for phone in tqdm(os.listdir(cfg.DATA_FOLDER)):
            embeddings[phone] = {}
            phone_path = os.path.join(cfg.DATA_FOLDER, phone)
            print(f"Phone path : {phone_path}")
            for file in os.listdir(phone_path):
                try:
                    file_path = os.path.join(phone_path, file)
                    filename = os.path.splitext(file)[0]
                    file_size = os.path.getsize(file_path)
                    # if > 20 MB skip
                    # if file_size > 20 * 1024 * 1024:
                    #     continue
                    print(f"\tFile path : {file_path}")
                    embeddings[phone][filename]=get_embedding(file_path,cfg.MODEL_NAME)
                    # print(embeddings[phone][filename])
                    all_wavs.append(file_path)
                except Exception as e:
                    print(f"Error in {file_path}: {e}")
                    continue
        os.makedirs(f"../../cache/{cfg.NAME}",exist_ok=True)
        np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",embeddings)
    print("Done!")
    # a = get_embedding("/home/zhaosheng/voiceprint-recognition-system/utils/dataset/cti_mini_test_8k/13055014477/cti_record_11003_1642640137635828_1-dcafce9b-5c47-47b3-8427-b35611fb464a.wav")
    # print(a.shape)