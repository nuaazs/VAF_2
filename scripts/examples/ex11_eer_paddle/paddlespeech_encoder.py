import torchaudio
import os
import subprocess
import cfg
from paddlespeech.cli.vector import VectorExecutor
vec = VectorExecutor()

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
    assert fs == 16000, "fs must be 16000"
    embeddings = vec(audio_file=output_filepath)
    return embeddings
# embeddings = classifier.encode_batch(signal)
