from tqdm import tqdm
import torch
from torch import nn
import dguard_nlp
from dguard_nlp.loss.margin_loss import get_projection
from dguard_nlp.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
import time

@dguard_nlp.utils.timeit
def train(config,model,optimizer,criterion,lr_scheduler,train_loader,vali_loader,test_loader,checkpointer,logger):
    device = config.device
    model.to(device)
    epochs = config.epochs
    valid_interval = config.valid_interval

    for epoch in range(epochs):
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
            prefix="Epoch: [{}]".format(epochs)
        )
        end = time.time()
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        for i,(data,labels) in enumerate(pbar):
            train_stats.update('Data', time.time() - end)
            iter_num = (epoch-1)*len(train_loader) + i
            # lr_scheduler.step(iter_num)
            # data.to(next(model.parameters()).device)
            labels = torch.tensor(labels).to(next(model.parameters()).device)
            # labels.to(next(model.parameters()).device)
            out=model(data) # [batch_size,num_class]
            loss=criterion.forward(out.to(device),labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if i%5==0:
            out=out.argmax(dim=-1)
            acc=(out.detach().cpu()==labels.detach().cpu()).sum().item()/len(labels.detach().cpu())
            # set tqdm prefix
            pbar.set_postfix(Loss=loss.item(),ACC=acc, refresh=False)        

            # recording
            train_stats.update('Loss', loss.item(), labels.size(0))
            train_stats.update('Acc@1', acc, labels.size(0))
            train_stats.update('Lr', optimizer.param_groups[0]["lr"])
            # train_stats.update('Margin', margin_scheduler.get_margin())
            train_stats.update('Time', time.time() - end)

            if i % config.log_batch_freq == 0:
                logger.info(progress.display(i))

            end = time.time()

        if epoch%valid_interval==0:
            # start valid
            checkpointer.save_checkpoint(epoch=epoch)
            model.eval()
            correct = 0
            total = 0
            for i,(data,labels) in enumerate(vali_loader):
                with torch.no_grad():
                    out=model(data)
                out = out.argmax(dim=1)
                correct += (out.cpu() == labels).sum().item()
                total += len(labels)
            acc_valid = correct / total

            correct = 0
            total = 0
            for i,(data,labels) in enumerate(test_loader):
                with torch.no_grad():
                    out=model(data)
                out = out.argmax(dim=1)
                correct += (out.cpu() == labels).sum().item()
                total += len(labels)
            acc_test = correct / total
            logger.info(f">>> Epoch {epoch} - Val ACC: {acc_valid} - Test ACC: {acc_test}")
            logger.info(f">>> Epoch {epoch} - Val ACC: {acc_valid} - Test ACC: {acc_test}")
    return model,optimizer