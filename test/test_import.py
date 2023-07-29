import importlib
import torch
import speakerlab.models as M
CKPT_PATH = {
    "CAMPP_EMB_512":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/campp_voxceleb/campp_voxceleb.bin",
    "ECAPA_TDNN_1024_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/ecapatdnn1024_voxceleb/ecapatdnn1024_voxceleb.bin",
    "ERES2NET_BASE_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/eres2netbase_voxceleb/eres2netbase_voxceleb.ckpt",
    "REPVGG_TINY_A0_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
}


if __name__ == '__main__':
    MODEL = "DFRESNET56_EMB_512"

    # import model
    model = getattr(M, MODEL)()

    # load parameters, set to eval mode, move to GPU
    device = "cuda:0"
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(CKPT_PATH[MODEL], map_location="cpu"),strict=True)

    # test inference
    with torch.no_grad():
        output = model(torch.randn(1, 200,80).to(device))
    print(output.shape)
    print(output)