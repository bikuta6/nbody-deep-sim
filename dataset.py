import json
import torch
from torch.utils.data import Dataset

class CollisionDataset(Dataset):
    def __init__(self, file_path, device='cpu'):
        with open(file_path) as f:
            self.data = json.load(f)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample_masses = torch.tensor(sample['masses'], dtype=torch.float32).unsqueeze(1).to(self.device)
        sample_pos0 = torch.tensor(sample['pos'], dtype=torch.float32).to(self.device)
        sample_vel0 = torch.tensor(sample['vel'], dtype=torch.float32).to(self.device)
        sample_pos1 = torch.tensor(sample['pos_next1'], dtype=torch.float32).to(self.device)
        sample_pos2 = torch.tensor(sample['pos_next2'], dtype=torch.float32).to(self.device)

        return sample_masses, sample_pos0, sample_vel0, sample_pos1, sample_pos2


def collate_fn(batch):
    # `batch` is a list of tensors
    return batch  # Return as-is to keep the varying sizes

def get_data_loader(file_path, batch_size=32, shuffle=True, device='cuda'):
    dataset = CollisionDataset(file_path, device=device)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)