from torch.utils.data import Dataset
class TCDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)