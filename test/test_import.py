import importlib
import torch
import dguard.models as M
from speakerlabduanyibo.utils.builder import build
from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.config import build_config
from speakerlabduanyibo.utils.fileio import load_wav_scp
from dguard.interface.pretrained import load_by_name,ALL_MODELS
CKPT_PATH = {
    "CAMPP_EMB_512":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/campp_voxceleb/campp_voxceleb.bin",
    "ECAPA_TDNN_1024_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/ecapatdnn1024_voxceleb/ecapatdnn1024_voxceleb.bin",
    "ERES2NET_BASE_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/eres2netbase_voxceleb/eres2netbase_voxceleb.ckpt",
    "REPVGG_TINY_A0_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
}


if __name__ == '__main__':
    #MODELs = ["resnet34_lm","resnet152_lm","resnet221_lm","resnet293_lm"] 
    MODELs = ["eres2net"]
    device = "cuda:0"
    # import model
    for MODEL in MODELs:
        model,feature_extractor,sample_rate = load_by_name(MODEL,device)
        # load parameters, set to eval mode, move to GPU
        model.eval()
        model.to(device)
        # test inference
        with torch.no_grad():
            output = model(torch.randn(1, 1440000,80).to(device))
        print(output.shape)
        print(output)
