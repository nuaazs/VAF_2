import requests
from tqdm import tqdm
import numpy as np

import requests
from tqdm import tqdm
import numpy as np
import cfg
import os

from utils import print_title

def get_embedding(file_path,model="ERES2NET",url="http://192.168.3.169:5001/encode",spkid="123456"):
    """get all wav's embedding, and save to npy file(each for a different model)
    

    Args:
        file_path (_type_): father path of all wav files
        model (str, optional): Split by <,>. Defaults to "ERES2NET".

    Returns:
        embedding(numpy array): _description_
    """
    payload={
        "spkid": spkid,
        "channel": 0,
        "filelist": f"local://{file_path}",
        "save_oss": "False",
        "score_threshold": 0.7,
        "start":"0",
        "length":"999",
    }
    # print(payload)
    response = requests.request("POST", url, data=payload)

    return_data = {}
    for _model_name in model.split(","):
        # print(response.json()["msg"])
        # print(_model_name)
        embeddings = response.json()["file_emb"][_model_name]["embedding"]
        temp_file = [_key for _key in embeddings.keys()][0]
        emb_result = embeddings[temp_file]
        return_data[_model_name] = np.array(emb_result)
    return return_data

if __name__ == "__main__":
    embeddings = {}
    all_wavs = []
    cache_path = f"../../cache/{cfg.NAME}"
    models = cfg.MODEL_NAME.split(",")
    all_embedding = {}
    if cfg.LOAD_NPY:
        #     for model in models:
        # print_title(model)
        # embedding_chache_file = f"{cache_path}/{model}/embeddings.npy" # 所有语音的embedding
        # # if npy exist, just load
        # if os.path.exists(embedding_chache_file) and cfg.LOAD_NPY:
        #     print(f"Loading {embedding_chache_file}...")
        #     embeddings = np.load(embedding_chache_file,allow_pickle=True).item()
        #     dict_info(embeddings)
        pass
    embeddings={}
    for model_name in models:
        embeddings[model_name]={}
    print("Generating embeddings...")
    for phone in tqdm(os.listdir(cfg.DATA_FOLDER)):
        for model_name in models:
            embeddings[model_name][phone] = {}

        phone_path = os.path.join(cfg.DATA_FOLDER, phone)
        print(f"Phone path : {phone_path}")
        for file in os.listdir(phone_path):
            try:
                file_path = os.path.join(phone_path, file)
                filename = os.path.splitext(file)[0]
                file_size = os.path.getsize(file_path)


                print(f"\tFile path : {file_path}")
                all_model_temp = get_embedding(file_path,cfg.MODEL_NAME,spkid=filename.replace("_","-"))
                for model_name in models:
                    embeddings[model_name][phone][filename] = all_model_temp[model_name]
                    print(f"\t\tModel name : {model_name}")
                    print(f"\t\t\tEmbedding shape : {all_model_temp[model_name].shape}")
            except Exception as e:
                print(f"Error in {file_path}: {e}")
                continue
    for model_name in models:
        os.makedirs(f"{cache_path}/{model_name}",exist_ok=True)
        np.save(os.path.join(f"{cache_path}/{model_name}","embeddings.npy"),embeddings)   
    sprint("Done!")