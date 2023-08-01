import torchaudio
from speechbrain.pretrained import EncoderClassifier
# import glob
import torch
import cfg
random_seed = torch.randint(1000, 9999, (1,)).item()
language_id = EncoderClassifier.from_hparams(source="./models/LANG", savedir=f"./pretrained_models/lang-id-ecapa", run_opts={"device":cfg.DEVICE})
language_id.eval()

def mandarin_filter(filelist,score_threshold=0.7):
    """
    Filter the mandarin audio
    """
    # read the wav file
    # waveform = wavdata.to(cfg.DEVICE)
    pass_list = []
    for filepath in filelist:
        wavdata = torchaudio.load(filepath)[0].reshape(1,-1).to(cfg.DEVICE)
        # change data_list to tensor, padding to the same length
        result = language_id.classify_batch(wavdata)
        score = result[1][0].exp()
        if score > score_threshold and result[3][0].startswith("zh"):
            pass_list.append(filelist[0])
        else:
            print(f"file: {filelist[0]} is not mandarin, result:{result[1][0]} {result[2][0]}")
    return pass_list

# if __name__ == '__main__':
    # mandarin_filter()