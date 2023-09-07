
from transformers import AdamW

def get_optimizer(config,model):
    learning_rate = config['lr']
    name = config['optimizer']
    if name == "AdamW":
        return AdamW(model.parameters(),lr=learning_rate)