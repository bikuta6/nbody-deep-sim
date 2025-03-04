import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
import pandas as pd
import os

class ParticleGraphDataset(InMemoryDataset):
    def __init__(self, csv_path, k=8, transform=None, pre_transform=None):
        self.csv_path = csv_path
        self.name = csv_path.split("/")[-1].split(".")[0]
        self.k = k
        self.root = os.path.dirname(csv_path)  # Required by InMemoryDataset
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.name}_graph.pt']  # Required by InMemoryDataset

    def process(self):
        df = pd.read_csv(self.csv_path)
        graphs = []

        for (scene, step), group in df.groupby(["scene", "step"]):
            positions = torch.tensor(group[["x", "y", "z"]].values, dtype=torch.float)
            velocities = torch.tensor(group[["vx", "vy", "vz"]].values, dtype=torch.float)
            accelerations = torch.tensor(group[["ax", "ay", "az"]].values, dtype=torch.float)
            mass = torch.tensor(group["mass"].values, dtype=torch.float).unsqueeze(1)

            edge_index = knn_graph(positions, k=self.k, loop=False)

            data = Data(
                x=torch.cat([positions, velocities, mass], dim=1),
                edge_index=edge_index,
                y=accelerations,
                scene=torch.tensor([scene]*len(positions)),
                step=torch.tensor([step]*len(positions))
            )
            graphs.append(data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

def get_dataloader(csv_path, batch_size=32, k=8, shuffle=True):
    dataset = ParticleGraphDataset(csv_path, k=k)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)