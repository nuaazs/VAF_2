import torchaudio
from speechbrain.pretrained import EncoderClassifier
# import glob
import torch
import cfg
random_seed = torch.randint(1000, 9999, (1,)).item()
language_id = EncoderClassifier.from_hparams(source="./nn/LANG", savedir=f"./pretrained_models/lang-id-ecapa_{random_seed}", run_opts={"device":cfg.DEVICE})
language_id.eval()

def filter_mandarin(wavdata,score_threshold=0.9):
    """
    Filter the mandarin audio
    """
    # read the wav file
    # waveform = wavdata.to(cfg.DEVICE)
    assert wavdata.device == torch.device("cuda:0")
    result = language_id.classify_batch(wavdata)
    score = result[1].exp()
    if score > score_threshold and result[3][0].startswith("zh"):
        return True,result[3][0],score
    else:
        return False,result[3][0],score

# if __name__ == "__main__":
#     root = "/lyxx/datasets/raw/fangyan/"
#     tn,tp,fn,fp = 0,0,0,0
#     wav_files = glob.glob(root + "**/*.wav", recursive=True)
#     for wav_file in wav_files:
#         waveform, sample_rate = torchaudio.load(wav_file)
#         result, lang, score = filter_mandarin(waveform,score_threshold=0.99)
#         # print(f"Source:{wav_file.split('/')[-2]} is mandarin: {result}. Language: {lang}. Score: {score}")
#         if not result and wav_file.split('/')[-2]=="cjsd":
#             print(f"Source:{wav_file.split('/')[-2]} is mandarin: {result}. Language: {lang}. Score: {score}")
#             print(f"# {wav_file}")
#             # os.remove(wav_file)
#             fn += 1
#         elif not result and wav_file.split('/')[-2]!="cjsd":
#             tn += 1
#         elif result and wav_file.split('/')[-2]=="cjsd":
#             tp += 1
#         elif result and wav_file.split('/')[-2]!="cjsd":
#             print(f"Source:{wav_file.split('/')[-2]} is mandarin: {result}. Language: {lang}. Score: {score}")
#             print(f"* {wav_file}")
#             fp += 1
#     print(f"tn:{tn},tp:{tp},fn:{fn},fp:{fp}")
#     print(f"precision:{tp/(tp+fp)},recall:{tp/(tp+fn)}")

        
