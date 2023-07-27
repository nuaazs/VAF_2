import torch
import torchaudio
import cfg


# model = pretrained.dns64().to(cfg.DEVICE)

def denoise_wav(wav_data):
    if len(wav_data.shape) == 1:
        wav_data = wav_data.unsqueeze(0)
    with torch.no_grad():
        result = model(wav_data.to(cfg.DEVICE))
        print(f"result shape: {result.shape}")
        denoised = result[0][0]
    return denoised


def denoise_file(filepath, savepath):
    # get wav data by torchaudio
    wav_data, sr = torchaudio.load(filepath)
    with torch.no_grad():
        denoised = model(wav_data.to(cfg.DEVICE))[0]
    # save audio by torchaudio
    torchaudio.save(savepath, denoised.detach().cpu(), sr)
