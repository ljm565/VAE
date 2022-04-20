from torch.utils.data import Dataset


class DLoader(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)


    def __getitem__(self, idx):
        return self.data[idx], -1

    
    def __len__(self):
        return self.length