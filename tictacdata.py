from torch.utils.data import Dataset
import torch

class TicTacDataset(Dataset):
    def __init__(self, correctEvals, states):
        self.labels = correctEvals
        self.states = states

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        state = self.states[idx]
        label = self.labels[idx]
        return torch.tensor(state), torch.tensor(label)
