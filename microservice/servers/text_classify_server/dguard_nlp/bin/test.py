from tqdm import tqdm
import torch
from torch import nn
import dguard_nlp
@dguard_nlp.utils.timeit
def test(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    for i,(data,labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out=model(data) # [batch_size,num_class]

        out = out.argmax(dim=1)
        # correct += (out.cpu() == labels).sum().item()
        # total += len(labels)
        tn =   ((out.cpu() == 0) & (labels == 0)).sum().item()
        tp =   ((out.cpu() == 1) & (labels == 1)).sum().item()
        fn =   ((out.cpu() == 0) & (labels == 1)).sum().item()
        fp =   ((out.cpu() == 1) & (labels == 0)).sum().item()
    acc_test = (tn+tp)/(tn+tp+fn+fp)
    recall_test = tp/(tp+fn)
    precision_test = tp/(tp+fp)
    # print(f">>> The accuracy of the model on the test set is: {correct / total * 100:.2f}% \n")
    print(f">>> The accuracy of the model on the test set is: {acc_test * 100:.2f}% \n")
    print(f">>> The recall of the model on the test set is: {recall_test * 100:.2f}% \n")
    print(f">>> The precision of the model on the test set is: {precision_test * 100:.2f}% \n")
    return correct / total * 100