# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import torch
import torchaudio
import importlib
from kaldiio import WriteHelper
from tqdm import tqdm
import dguard.models as M
from speakerlabduanyibo.utils.builder import build
from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.config import build_config
from speakerlabduanyibo.utils.fileio import load_wav_scp
from dguard.interface.pretrained import load_by_name,ALL_MODELS

parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
parser.add_argument('--exp_dir', default='', type=str, help='Exp dir')
parser.add_argument('--data', default='', type=str, help='Data dir')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
CKPT_PATH = {
    "CAMPP_EMB_512":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/campp_voxceleb/campp_voxceleb.bin",
    "ECAPA_TDNN_1024_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/ecapatdnn1024_voxceleb/ecapatdnn1024_voxceleb.bin",
    "ERES2NET_BASE_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/eres2netbase_voxceleb/eres2netbase_voxceleb.ckpt",
    "REPVGG_TINY_A0_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_95":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-95-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_95":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-95-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_90":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-90-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_90":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-90-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_85":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-85-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_85":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-85-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_80":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-80-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_80":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-80-00/embedding_model.ckpt",
}





def calculate_cosine_distance(x, y):
    """计算余弦距离"""
    # x: [batch_size, embedding_size]
    # y: [batch_size, embedding_size]
    # output: [batch_size]
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.sum(x * y, dim=1)

def calculate_cmf(embeddings):
    """计算 CMF"""
    print(f"recieved embeddings shape: {embeddings.shape}")
    # input embeddings: [N,batch_size,embedding_size]
    # output cmf : (batch_size,)
    
    # Normalize the embeddings along the embedding_size dimension (dim=2)
    embeddings_normalized = F.normalize(embeddings, p=2, dim=2)
    
    cmf = torch.mean(embeddings_normalized, dim=0)
    print(f"cmf shape: {cmf.shape}")
    return cmf
    

def cmf_score_calibration(embA, embB, CMFA, CMFB):
    """CMF 分数校准"""
    # CMFA: [batch_size, embedding_size]
    # CMFB: [batch_size, embedding_size]
    # embA: [batch_size, embedding_size]
    # embB: [batch_size, embedding_size]
    cos_score = calculate_cosine_distance(embA, embB)
    # cos_score: [batch_size]
    # CMFA * CMFB: [batch_size, embedding_size]
    # from IPython import embed; embed()
    N = CMFA.shape[0]
    result = []
    for i in range(N):
        factor = torch.dot(CMFA[i],CMFB[i])
        score =  factor * cos_score[i]
        print(f"score: {score}, factor: {factor}, cos_score: {cos_score[i]}")
        result.append(score)
    result = torch.stack(result, dim=0)
    return result

def random_choose_ten_crops(wav_data, segment_length,get_embedding_func):
    # When segmenting audio to calculate CMF, there is an overlap
    # of half the segment length so that the segment embedding do
    # not diverge too much.
    # wav_data: [batch_size, T]
    batch_size, T = wav_data.shape
    
    # Calculate the number of segments
    num_segments = T // (segment_length // 2) - 1
    
    # Initialize a list to store the selected crops
    selected_crops = []
    selected_crops_emb = []
    
    for i in range(num_segments):
        # Calculate the start and end indices of the segment
        start_idx = i * (segment_length // 2)
        end_idx = start_idx + segment_length
        
        # Extract the segment from the waveform data
        segment = wav_data[:, start_idx:end_idx]
        emb = get_embedding_func(segment)
        # Append the segment to the selected crops
        selected_crops.append(segment)
        selected_crops_emb.append(emb)
        # Break the loop if we have selected ten crops
        # if len(selected_crops) == 20:
        #     break
    
    # Stack the selected crops along the batch dimension
    selected_crops = torch.stack(selected_crops, dim=0)
    selected_crops_emb = torch.stack(selected_crops_emb, dim=0)
    return selected_crops,selected_crops_emb

def get_cmf(wav_data, segment_length,get_embedding_func):
    selected_crops_a,embeddings_a = random_choose_ten_crops(wav_data, segment_length,get_embedding_func)
    cmf = calculate_cmf(embeddings_a)
    return cmf

# if __name__ == '__main__':
#     # 示例调用
#     # 16000 * 20
#     wav_data_a,sr = torchaudio.load("/datasets/cjsd_download_test_vad/female_13/s2023_07_31_20_16_45_e2023_07_31_20_17_31.wav") # torch.randn(6, 320000)  # 音频数据，形状为 [batch_size, T]
#     wav_data_b,sr = torchaudio.load("/datasets/cjsd_download_test_vad/female_13/s2023_07_31_20_18_27_e2023_07_31_20_19_33.wav")   # 音频数据，形状为 [batch_size, T]
#     # 1/s2023_07_31_18_41_19_e2023_07_31_18_42_17.wav") #
#     segment_length = 3*16000  # 分段长度

#     selected_crops_a,embeddings_a = random_choose_ten_crops(wav_data_a, segment_length,get_embedding_func=get_embedding) # [N,batch_size,segment_length]
#     selected_crops_b,embeddings_b = random_choose_ten_crops(wav_data_b, segment_length,get_embedding_func=get_embedding) # [N,batch_size,segment_length]
#     print(f"split A -> #({selected_crops_a.shape})")  # 输出选取的十个分段的形状
#     print(f"embeddings_a: {embeddings_a.shape}")
#     print(f"embeddings_b: {embeddings_b.shape}")

#     embedding_a = get_embedding(wav_data_a)  # 提取音频 A 的嵌入向量
#     print(f"embedding_a: {embedding_a.shape}")
#     embedding_b = get_embedding(wav_data_b)  # 提取音频 B 的嵌入向量
#     print(f"embedding_b: {embedding_b.shape}")
#     CMF_A = calculate_cmf(embeddings_a)  # 计算音频 A 的 CMF 值
#     print(f"CMF_A: {CMF_A}")
#     print(f"CMF_A shape: {CMF_A.shape}")
#     CMF_B = calculate_cmf(embeddings_b)  # 计算音频 B 的 CMF 值
#     print(f"CMF_B: {CMF_B}")
#     print(f"CMF_B shape: {CMF_B.shape}")
#     score_a_b = cmf_score_calibration(embedding_a, embedding_b, CMF_A, CMF_B)  # 根据 CMF 进行分数校准
#     print(f"score_a_b: {score_a_b}")

def main():
    args = parser.parse_args(sys.argv[1:])
    config_file = os.path.join("/home/duanyibo/dyb/test_model/speakerlabduanyibo", 'config.yaml')
    config = build_config(config_file)
    # rank = 0
    # world_size = 1

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # assert world_size == 7, f" world_size {world_size} != 7 "

    embedding_dir = os.path.join(args.exp_dir, 'embeddings')
    os.makedirs(embedding_dir, exist_ok=True)

    logger = get_logger()

    if args.use_gpu:
        if torch.cuda.is_available():
            gpu = int(args.gpu[rank % len(args.gpu)])
            device = torch.device('cuda', gpu)
        else:
            msg = 'No cuda device is detected. Using the cpu device.'
            if rank == 0:
                logger.warning(msg)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    MODEL = args.exp_dir.split("/")[-2]

    # import model
    # model = getattr(M, MODEL)()

    # load parameters, set to eval mode, move to GPU
    # device = "cuda:0"

    # model.load_state_dict(torch.load(CKPT_PATH[MODEL], map_location="cpu"),strict=True)

    # Build the embedding model
    # feature_extractor = build('feature_extractor', config)
    model,feature_extractor,sample_rate = load_by_name(MODEL,device)
    model.eval()
    model.to(device)
    print(args.data)
    data = load_wav_scp(args.data)
    data_k = list(data.keys())
    local_k = data_k[rank::world_size]
    if len(local_k) == 0:
        msg = "The number of threads exceeds the number of files"
        logger.info(msg)
        sys.exit()

    emb_ark = os.path.join(embedding_dir, 'xvector_%02d.ark'%rank)
    emb_scp = os.path.join(embedding_dir, 'xvector_%02d.scp'%rank)
    emb_cmf = os.path.join(embedding_dir, 'xvector_%02d.cmf'%rank)
    if rank == 0:
        logger.info('Start extracting embeddings.')
    with torch.no_grad():
        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp},{emb_cmf}') as writer:
            for k in tqdm(local_k):
                wav_path = data[k]
                if args.exp_dir.split("/")[-1] == "cti_result":
                    num_samples = int(10*config.sample_rate)
                    wav, fs = torchaudio.load(wav_path, frame_offset=0, num_frames=num_samples)
                else:
                    wav, fs = torchaudio.load(wav_path)
                assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
                feat = feature_extractor(wav)
                feat = feat.unsqueeze(0)
                feat = feat.to(device)
                cmf = get_cmf(wav, 3*16000,get_embedding_func=lambda x: model(x)[-1].detach().cpu().numpy())
                emb = model(feat)[-1].detach().cpu().numpy()

                # emb = mode(feat).detach().cpu().numpy()
                writer(k, emb)

if __name__ == "__main__":
    main()
