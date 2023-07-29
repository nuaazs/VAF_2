# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from speakerlab.utils.builder import dynamic_import
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument('--model_id', default='damo/speech_campplus_sv_zh-cn_16k-common', help='from 3dspeaker.')
parser.add_argument('--model_stage', default='True', help='True: Update parameters False:Freeze the model')

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}
ERes2Net_common = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k':{
        'revision': 'v1.0.0',
        'model': ERes2Net_common,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k':{
        'revision': 'v1.0.0',
        'model': ERes2Net_common,
        'model_pt': 'eres2net_base_model.ckpt',
    }
    # task='speaker-verification',
    # model='damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
    # model_revision='v1.0.0'
}
def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)
    print(args.model_stage,"#"*100)
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Use GPU: {gpu} for training.")

    # dataset
    train_dataset = build('dataset', config)
    # dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.dataloader['args']['sampler'] = train_sampler
    config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    train_dataloader = build('dataloader', config)

    # model
    # print(bool(args.model_stage))
    if args.model_stage == "True":
        embedding_model = build('embedding_model', config) #初始化训练模型
        
        # N=0
        # for param in embedding_model.parameters():
        #     N+=1
        #     param.requires_grad = False
        #     if N==571:
        #         param.requires_grad = True
        #     print(param.requires_grad)
    else:
    ##########读取保存的预训练模型##########
        print("aaaaaaaaaa"*10)
        conf = supports[args.model_id]
        para =torch.load('/home/duanyibo/dyb/3dspeaker/3D-Speaker/pretrained/speech_campplus_sv_zh-cn_16k-common/CAMPP.pth')
        
        model = conf['model']  
        #支持直接下载预训练模型/home/duanyibo/dyb/3dspeaker/3D-Speaker/speakerlab/bin/infer_sv.py
        # python speakerlab/bin/infer_sv.py --model_id damo/speech_campplus_sv_zh-cn_16k-common --wavs /home/duanyibo/dyb/3dspeaker/3D-Speaker/pretrained/speech_campplus_sv_zh-cn_16k-common/exm/18659111928_001_Distance00_Dialect00.wav /home/duanyibo/dyb/3dspeaker/3D-Speaker/pretrained/speech_campplus_sv_zh-cn_16k-common/exm/19513386018_001_Distance00_Dialect00.wav
        # 预训练model_id
        # ['damo/speech_campplus_sv_en_voxceleb_16k' ,
        # 'damo/speech_campplus_sv_zh-cn_16k-common',
        # 'damo/speech_eres2net_sv_en_voxceleb_16k',
        # 'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
        # 'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k']

        #￥￥￥模型加载参数￥￥￥
        embedding_model = dynamic_import(model['obj'])(**model['args'])
        embedding_model.load_state_dict(para)
        # 停止梯度更新
        # embedding_model.eval()
        for param in embedding_model.parameters():
            param.requires_grad = False
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)
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

    # resume from a checkpoint
    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    cudnn.benchmark = True

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        # train one epoch
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
        )

        if rank == 0:
            # log
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        dist.barrier()

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank):
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

    #train mode
    model.train()
    for name,param in model.named_parameters():
        # print(name)
        # if "module.0.xvector.dense.linear.weight" in name or"module.1.weight" in name :
        #     param.requires_grad = True
        # else:
        #     param.requires_grad = False
        print(param.requires_grad)
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        # update
        iter_num = (epoch-1)*len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # compute output
        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracy(output, y)

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
