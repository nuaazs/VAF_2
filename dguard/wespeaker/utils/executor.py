# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torchnet as tnt


def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              logger,
              scaler,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda'),
              base_feature_extractor=None,
              base_model=None):
    model.train()
    base_model.to(device)
    base_model.eval()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    for i, batch in enumerate(dataloader):
        utts = batch['key']
        targets = batch['label']
        if "wav_data" not in batch:
            features = batch['feat']
        else:
            if base_feature_extractor is not None:
                wav_data = batch['wav_data']
                sample_rate = 16000 # TODO dynamic sample rate
                # print(wav_data)
                wav_data.to(device)
                # print(f"wav_data shape: {wav_data.shape}")
                wav_data = wav_data.reshape(wav_data.shape[0],-1)
                with torch.no_grad():
                    input_values = base_feature_extractor(wav_data, return_tensors="pt",sampling_rate=sample_rate).input_values
                    input_values = input_values.squeeze(0)
                    # print(f"input_values shape: {input_values.shape}")
                    with torch.no_grad():
                        input_values = input_values.to(device)
                        outputs = base_model(input_values)
                        features = outputs.last_hidden_state
                # print(f"Final features shape: {features.shape}")
                # features = features[0]
            else:
                raise ValueError("base_feature_extractor is None and wav_data is in batch !!")

        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        features = features.float().to(device)  # (B,T,F)
        targets = targets.long().to(device)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % log_batch_interval == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
