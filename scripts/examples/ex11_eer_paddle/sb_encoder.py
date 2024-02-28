import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os
import subprocess
import cfg
# load model to cuda:0
classifier = EncoderClassifier.from_hparams(source=cfg.MODEL_PATH,run_opts={"device": "cuda:0"})
# signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
#print(f"Load model: {classifier}")
def generate_embedding(file_path):
    # resample file to 16k
    output_filepath = file_path.replace('8k', '16k')
    # mkdir output_dir
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # resample
    if not os.path.exists(output_filepath):
        cmd = f"ffmpeg -i {file_path} -ar 16000 {output_filepath}"
        subprocess.call(cmd, shell=True)
    signal, fs =torchaudio.load(output_filepath)
    embeddings = classifier.encode_batch(signal)
    return embeddings
# embeddings = classifier.encode_batch(signal)
