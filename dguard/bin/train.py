# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import re
import os
import sys
import time
import torch
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from kaldiio import ReadHelper
from kaldiio import WriteHelper
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from sklearn.metrics.pairwise import cosine_similarity

# Dguard utils
from dguard.utils.builder import build,try_build_component
from dguard.utils.fileio import load_wav_scp
from dguard.utils.config import build_config
from dguard.utils.epoch import EpochCounter, EpochLogger
from dguard.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from dguard.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)

parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpus', nargs='+', help='GPU id to use.')
parser.add_argument('--origin_model_dir', default='', type=str, help='Checkpoint directory of the original model')
parser.add_argument('--epoch_classify', default=60, type=int, help='Number of epochs for classifier training')
parser.add_argument('--epoch_all', default=60, type=int, help='Number of epochs for whole model training')

def conditional_print(func):
    def wrapper(*args, **kwargs):
        if rank == 0:
            result = func(*args, **kwargs)
        else:
            result = None
    return wrapper

@conditional_print
def conditional_logger(logger,message):
    logger.info(f"#=> System: {message}")

def main():
    # parse arguments initialize
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpus[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')
    set_seed(args.seed+rank)
    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir, terminal=True)
    logger.info(f"#=> Use GPU: #{gpu} for training ...")
    conditional_logger(logger, f"##=====> Initialize training <=====##")
    conditional_logger(logger, f"World size: {world_size}, Rank: {rank}")
    conditional_logger(logger, f"EXP DIR: {config.exp_dir}")
    for k, v in config.items():
        conditional_logger(logger, f"{k}: {v}")

    # dataset
    train_dataset = build('dataset', config)

    # dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.dataloader['args']['sampler'] = train_sampler
    config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    train_dataloader = build('dataloader', config)
    conditional_logger(logger, f"##=====> Data Set <=====##")
    conditional_logger(logger, f"Train dataset num: {len(train_dataset)}")

    # embedding model
    embedding_model = build('embedding_model', config)
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * len(config.speed_pertub)
    else:
        config.num_classes = len(config.label_encoder)
    freeze_embedding = config.get('freeze_embedding', False)
    conditional_logger(logger, f"##=====> Embedding model <=====##")
    num_params = sum(param.numel() for param in embedding_model.parameters())
    conditional_logger(logger, f"Number of parameters: {num_params}")
    conditional_logger(logger, f"Freeze embedding: {freeze_embedding}")

    # classifier model
    classifier = build('classifier', config)
    feature_extractor = try_build_component('feature_extractor', config, logger)
    pre_extractor = try_build_component('pre_extractor', config, logger)
    conditional_logger(logger, f"##=====> Classifier model <=====##")
    num_params = sum(param.numel() for param in classifier.parameters())
    conditional_logger(logger, f"Number of parameters: {num_params}")
    conditional_logger(logger, f"Pre-extractor: {pre_extractor}")
    conditional_logger(logger, f"Feature-extractor: {feature_extractor}")

    if freeze_embedding:
        for param in embedding_model.parameters():
            param.requires_grad = False
        logger.info("#=> Embedding model is frozen.")

    model = nn.Sequential(embedding_model, classifier)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    # optimizer
    config.optimizer['args']['params'] = model.parameters()
    optimizer = build('optimizer', config)

    # loss function
    criterion = build('loss', config)

    # scheduler
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)
    
    # others
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    conditional_logger(logger, f"##=====> Start training <=====##")
    conditional_logger(logger, f"Optimizer: {optimizer}")
    conditional_logger(logger, f"Criterion: {criterion}")
    conditional_logger(logger, f"LR Scheduler: {lr_scheduler}")
    conditional_logger(logger, f"Margin Scheduler: {margin_scheduler}")
    conditional_logger(logger, f"Epoch Log file: {os.path.join(config.exp_dir, 'train_epoch.log')}")
    conditional_logger(logger, f"Main log file: {logger.handlers[0].baseFilename}")

    if args.resume:
        checkpointer.recover_if_possible(device='cuda')
    cudnn.benchmark = True

    # start training
    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)
        train_stats = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            lr_scheduler,
            margin_scheduler,
            logger,
            config,
            rank,
            pre_extractor
        )
        if rank == 0:
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)
                logger.info(f"#=> Save checkpoint at epoch {epoch}.")
                
        try:
            if epoch % config.save_epoch_freq== 0:
                train_stats["eer_test"] = f"{eer:.3f}"
                train_stats["min_dcf_test"] = f"{min_dcf:.3f}"
                train_stats["min_dcf_noc_test"] = f"{min_dcf_noc:.3f}"
        except Exception as e:
            logger.error(e)
        dist.barrier()

def test(model, gpu,config, epoch, logger, rank, wav_scp,embedding_dir,feature_extractor=None,pre_extractor=None):
    """Test the model

    Args:
        model (nn.Module): model
        gpu (int): gpu id
        config (dict): config
        epoch (int): epoch number
        logger (logging.Logger): logger
        rank (int): rank
        wav_scp (str): wav.scp file
        embedding_dir (str): embedding directory
        feature_extractor (nn.Module): feature extractor
        pre_extractor (nn.Module): pre extractor

    Returns:
        dict: key stats
    """
    model.eval()
    with torch.no_grad():
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        data = load_wav_scp(wav_scp)
        data_k = list(data.keys())
        tiny_len = int( len(data_k) / world_size)
        local_k = data_k[rank * tiny_len : (rank+1) * tiny_len]
        if rank == world_size - 1:
            local_k = data_k[rank * tiny_len : ]
        emb_ark = os.path.join(embedding_dir, 'xvector_%02d.ark'%rank)
        emb_scp = os.path.join(embedding_dir, 'xvector_%02d.scp'%rank)
        if rank == 0:
            logger.info('#=> Start extracting embeddings.')
            with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
                for k in local_k:
                    wav_path = data[k]
                    wav, fs = torchaudio.load(wav_path)
                    if feature_extractor:
                        feat = feature_extractor(wav)
                    else:
                        feat = wav
                    if pre_extractor:
                        feat = pre_extractor(feat)
                    if len(feat.shape) == 2:
                        feat = feat.unsqueeze(0)
                    feat = feat.to(gpu)
                    outputs = model(feat)
                    logger.info(f"#=> Test: model output shape: {outputs.shape}")
                    embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                    emb = embeds.detach().cpu().numpy()
                    writer(k, emb)
    return emb_ark,emb_scp

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank, feature_extractor, pre_extractor):
    """Train for one epoch on the training set

    Args:
        train_loader (torch.utils.data.DataLoader): training set
        model (nn.Module): model
        criterion (nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): epoch number
        lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler
        margin_scheduler (torch.optim.lr_scheduler): margin scheduler
        logger (logging.Logger): logger
        config (dict): config
        rank (int): rank
        feature_extractor (nn.Module): feature extractor
        pre_extractor (nn.Module): pre extractor
    
    Returns:
        dict: key stats
    """
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        train_stats.update('Data', time.time() - end)
        iter_num = (epoch-1)*len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        if feature_extractor:
            x = feature_extractor(x)
        if pre_extractor:
            x = pre_extractor(x)
        outputs = model(x)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        loss = criterion(embeds, y)
        acc1 = accuracy(embeds, y)
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # recording
        train_stats.update('Loss', loss.item(), x.size(0))
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)
        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))
        end = time.time()
    key_stats={
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats

if __name__ == '__main__':
    main()