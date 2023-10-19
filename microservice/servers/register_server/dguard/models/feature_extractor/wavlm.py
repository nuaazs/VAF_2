
import torch
from torchaudio.sox_effects import apply_effects_file
from dguard.models.feature_extractor.wavlm_models import WavLM, WavLMConfig
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
class wavlm_extractor(torch.nn.Module):
    def __init__(self, checkpoint="/VAF/train/pretrained_models/WavLM-Large.pt",device="cuda"):
        super(wavlm_extractor, self).__init__()
        self.device = device
        ckpt = torch.load(checkpoint)
        self.cfg = WavLMConfig(ckpt['cfg'])
        print(f"Loading WavLM-Large cfg success")
        self.model = WavLM(self.cfg)
        print(f"Loading WavLM-Large success")
        self.model.load_state_dict(ckpt['model'])
        print(f"Loading WavLM-Large state_dict success")
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()

    def forward(self, input_values):
        # send input_values to model's device
        input_values = input_values.to(next(self.model.parameters()).device)
        input_values = input_values.reshape([input_values.shape[0],-1])
        # input_values = input_values.to(self.device)
        if self.cfg.normalize:
            input_values = torch.nn.functional.layer_norm(input_values , input_values.shape)
        feature,mask = self.model.extract_features(input_values)
        # feature = feature.squeeze(0)
        return feature

if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    model = wavlm_extractor(checkpoint="/VAF/train/pretrained_models/WavLM-Large.pt")
    model.to("cuda:7")
    model.eval()
    print(f"Model Size: {sum(p.numel() for p in model.parameters())}")
    # print trainable parameters
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    EFFECTS = [
        ["remix", "-"],
        ["channels", "1"],
        ["rate", "16000"],
        ["gain", "-1.0"],
        ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
        ["trim", "0", "10"],
    ]
    input_values = apply_effects_file("/VAF/train/data/raw_data/voxceleb1/dev/wav/id10001/7gWzIy6yIIk/00002.wav", EFFECTS)
    input_values = torch.tensor(input_values[0].numpy())
    feature = model(input_values)
    print(feature.shape)
    print(feature)