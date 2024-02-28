import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os
import subprocess
import cfg
# load model to cuda:0
print(f"Model Path:{cfg.MODEL_PATH}")
classifier = EncoderClassifier.from_hparams(source=cfg.MODEL_PATH,run_opts={"device": "cuda:0"})

def generate_embedding(file_path,sr=16000):
    # resample file to sr
    # temp dir
    # temp_dir = "./temp"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # output_filepath = os.path.join(temp_dir,os.path.basename(file_path))
    # # resample
    # if not os.path.exists(output_filepath):
    #     cmd = f"ffmpeg -i {file_path} -ar {sr} {output_filepath} -y > /dev/null 2>&1"
    #     subprocess.call(cmd, shell=True)
    signal, fs =torchaudio.load(file_path)
    assert fs == sr, f"Sample rate of {file_path} is {fs}, not {sr}"
    embeddings = classifier.encode_batch(signal)
    # rm temp file
    # os.remove(output_filepath)
    # print(f"Get embedding of {file_path}, SR={sr}, Model={cfg.MODEL_PATH}")
    return embeddings
