# coding = utf-8
# @Time    : 2023-08-29  08:45:50
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: CMF (Consistency Measure Factor score calibration).

import torch
import torch.nn.functional as F
import torchaudio

# import sys
# sys.path.append("/VAF/train")
# from dguard.interface.pretrained import load_by_name,ALL_MODELS
# print(ALL_MODELS)
# model,feature_extractor,sample_rate = load_by_name('repvgg',strict=False)
# model.eval()


# def get_embedding(wav_data,embedding_size=256):
#     # wav_data: [batch_size, T]
#     # NOTE: just return fake embedding now. You should implement your own embedding extractor.
#     batch_size, T = wav_data.shape
#     wav = torch.tensor(wav_data, dtype=torch.float32)
#     feat = feature_extractor(wav)
#     feat = feat.unsqueeze(0)
#     feat = feat.to(next(model.parameters()).device)
#     with torch.no_grad():
#         outputs = model(feat)
#         # outputs = model(x)
#         embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
#         output = embeds.detach()#.cpu().numpy()
#     return output

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
    # TODO: 不支持批量计算
    # 1. 将每个embedding归一化
    # 2. 计算sum_embedding
    # 3. 将sum_embedding归一化
    # 4. 计算归一化后的sum_embedding的长度
    data = []
    for tiny_data in embeddings:
        tiny_data = tiny_data.reshape(-1)
        # _min = torch.min(tiny_data)
        # _max = torch.max(tiny_data)
        # tiny_data = (tiny_data - _min) / (_max - _min)
        _len = (tiny_data**2).sum().sqrt()#/(tiny_data.shape[0]**0.5)
        tiny_data = tiny_data/_len
        data.append(tiny_data.reshape(1,-1))
    data = torch.cat(data, dim=0)
    print(f"data: {data.shape}")
    sum_embedding = torch.sum(data, dim=0).reshape(-1)
    print(f"sum_embedding: {sum_embedding}")
    # _min = torch.min(sum_embedding)
    # _max = torch.max(sum_embedding)
    # sum_embedding = (sum_embedding - _min) / (_max - _min)
    length = (sum_embedding**2).sum().sqrt()#/(sum_embedding.shape[0]**0.5)
    length = float(length)/data.shape[0]
    print(length)
    return length

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
        # print(f"score: {score}, factor: {factor}, cos_score: {cos_score[i]}")
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
    # print(f"num_segments: {num_segments}")
    # Initialize a list to store the selected crops
    selected_crops = []
    selected_crops_emb = []
    
    for i in range(num_segments):
        # Calculate the start and end indices of the segment
        start_idx = i * (segment_length // 2)
        end_idx = start_idx + segment_length
        
        # Extract the segment from the waveform data
        segment = wav_data[:, start_idx:end_idx]
        # print(f"segment shape: {segment.shape}")
        emb = get_embedding_func(segment)
        # print(f"emb shape: {emb.shape}")
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