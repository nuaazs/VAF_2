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

from dguard.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from dguard.utils.config import build_config
from dguard.utils.builder import build
from dguard.utils.epoch import EpochCounter, EpochLogger

from sklearn.metrics.pairwise import cosine_similarity
###############################################################################################################
from dguard.utils.utils import get_logger
from dguard.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)
from dguard.utils.builder import build
from dguard.utils.utils import get_logger
from dguard.utils.config import build_config
from dguard.utils.fileio import load_wav_scp
from kaldiio import WriteHelper
import torchaudio
import re
import os
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import argparse
import torch
import torchaudio
from kaldiio import WriteHelper

from dguard.utils.builder import build
from dguard.utils.utils import get_logger
from dguard.utils.config import build_config
from dguard.utils.fileio import load_wav_scp
###############################################################################################################
from IPython import embed

test_config = {
    "trials":["/datasets/voxceleb1/trials/vox1_O_cleaned.trial"], # ,"/datasets/voxceleb1/trials/vox1_E_cleaned.trial","/datasets/voxceleb1/trials/vox1_H_cleaned.trial"
    "test_epoch_freq":1,
    "wav_scp":"/datasets/voxceleb1/test/wav/wav.scp" # # /home/duanyibo/dyb/test_model/voxceleb1
}

parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

# TODO: fine-tune 功能实现
parser.add_argument('--fine_tune', action='store_true', help='Whether to perform fine-tuning')
parser.add_argument('--origin_model_dir', default='', type=str, help='Checkpoint directory of the original model')
parser.add_argument('--epoch_classify', default=60, type=int, help='Number of epochs for classifier training')
parser.add_argument('--epoch_all', default=60, type=int, help='Number of epochs for whole model training')

def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

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
    embedding_model = build('embedding_model', config)
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)

    feature_extractor = build('feature_extractor', config)

    
    if args.fine_tune:
        if args.origin_model_dir:
            # Load the parameter of the original model
            checkpoint = torch.load(args.origin_model_dir)
            model.load_state_dict(checkpoint['state_dict'])

        # Freeze the embedding_model
        for param in embedding_model.parameters():
            param.requires_grad = False

        # Reinitialize the classifier
        classifier.reset_parameters()
        model = nn.Sequential(embedding_model, classifier)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
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
    # embed()

    for epoch in epoch_counter:
        

        train_sampler.set_epoch(epoch)

        # train one epoch
        if args.fine_tune:
            train_stats = train_fine_tune(
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
                epoch_classify=args.epoch_classify,
                epoch_all=args.epoch_all
            )
        else:

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


        #####################################################################################
        try:
            # test voxceleb1-O voxceleb1-H voxceleb1-E
            if epoch % config.save_epoch_freq== 0:

                embedding_dir = os.path.join(config.exp_dir, 'embeddings')
                os.makedirs(embedding_dir, exist_ok=True)
                # rm all ark and scp files in embedding_dir
            
                emb_ark,emb_scp = test(model, int(args.gpu[rank]),test_config, epoch, logger, rank,test_config["wav_scp"],embedding_dir=embedding_dir,feature_extractor=feature_extractor)
            # torchrun 等待其他程序运行完成
            dist.barrier()
            if epoch % config.save_epoch_freq and rank == 0:
                logger.info('Finish epoch %d.' % epoch)
                logger.info('Compute eer')
                # compute eer
                # get emb_ark father dir 
                endol_data_dir = os.path.join(config.exp_dir, 'embeddings')
                test_data_dir = endol_data_dir
                scores_dir = os.path.join(config.exp_dir, 'scores')
                eer,min_dcf,min_dcf_noc = get_eer(logger,endol_data_dir,test_data_dir,scores_dir,test_config["trials"])
                # train_stats.add('eer_test', ':6.3f')
                # train_stats.update('eer_test', eer)
                # train_stats.add('min_dcf_test', ':6.3f')
                # train_stats.update('min_dcf_test', min_dcf)
                # train_stats.add('min_dcf_noc_test', ':6.3f')
                # train_stats.update('min_dcf_noc_test', min_dcf_noc)
                train_stats["eer_test"] = f"{eer:.3f}"
                train_stats["min_dcf_test"] = f"{min_dcf:.3f}"
                train_stats["min_dcf_noc_test"] = f"{min_dcf_noc:.3f}"

        except Exception as e:
            logger.info(e)
            print("Error in test")
            print(e)
        #####################################################################################

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


def test(model, gpu,config, epoch, logger, rank, wav_scp,embedding_dir,feature_extractor):
    model.eval()
    # no grad
    with torch.no_grad():
        # extract embeddings
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # gpu = int(args.gpu[rank])
        data = load_wav_scp(wav_scp)
        data_k = list(data.keys())
        tiny_len = int( len(data_k) / world_size)
        local_k = data_k[rank * tiny_len : (rank+1) * tiny_len]
        if rank == world_size - 1:
            local_k = data_k[rank * tiny_len : ]
        # local_k = data_k[rank::world_size]
        # print(len(local_k))
        # print(len(data_k))
        

        emb_ark = os.path.join(embedding_dir, 'xvector_%02d.ark'%rank)
        emb_scp = os.path.join(embedding_dir, 'xvector_%02d.scp'%rank)
        
        if rank == 0:
            logger.info('Start extracting embeddings.')
        with torch.no_grad():
            with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
                for k in local_k:
                    wav_path = data[k]
                    wav, fs = torchaudio.load(wav_path)
                    feat = feature_extractor(wav)
                    feat = feat.unsqueeze(0)
                    feat = feat.to(gpu)
                    emb = model(feat).detach().cpu().numpy()
                    writer(k, emb)
    return emb_ark,emb_scp

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


def train_fine_tune(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank,epoch_classify=0,epoch_all=0):
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
        if epoch == epoch_classify:
            # After training the classifier for epoch_classify epochs, unfreeze the embedding_model
            for param in model.module.embedding_model.parameters():
                param.requires_grad = True

    key_stats={
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats

def get_eer(logger,endol_data_dir,test_data_dir,scores_dir,trials,p_target=0.01,c_miss=1,c_fa=1):

    os.makedirs(scores_dir, exist_ok=True)

    result_path = os.path.join(scores_dir, 'result.metrics')
    if os.path.exists(result_path):
        # rm
        os.remove(result_path)

    def collect(data_dir):
        data_dict = {}
        emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i)]
        if len(emb_arks) == 0:
            raise Exception(f'No embedding ark files found in {data_dir}')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict

    enrol_dict = collect(endol_data_dir)
    test_dict = collect(test_data_dir)

    for trial in trials:
        scores = []
        labels = []

        trial_name = trial.split('/')[-1].split('.')[0]
        # get 
        score_path = os.path.join(scores_dir, f'{trial_name}.score')
        with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
            lines = trial_f.readlines()
            for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                pair = line.strip().split()
                enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                              test_emb.reshape(1, -1))[0][0]
                # write the score
                score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                scores.append(cosine_score)
                if pair[2] == '1' or pair[2] == 'target':
                    labels.append(1)
                elif pair[2] == '0' or pair[2] == 'nontarget':
                    labels.append(0)
                else:
                    raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        labels = np.array(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=p_target,
                                c_miss=c_miss,
                                c_fa=c_fa)
        min_dcf_noc = compute_c_norm(fnr,
                                fpr,
                                p_target=0.000005,
                                c_miss=1,
                                c_fa=5)

        # write the metrics
        logger.info("Results of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            p_target, c_miss, c_fa, min_dcf))
        print(f"EER = {100 * eer:.4f}")
        return eer,min_dcf,min_dcf_noc

if __name__ == '__main__':
    main()
