import tqdm
from generation import NBodyDataset
from torch.utils.data import DataLoader
import torch
import os
from datetime import datetime


class Trainer:
    def __init__(self, model, optimizer, edge_attr, acc_weight, force_weight, neighbor_weight, energy_weight, device):
        self.model = model
        self.optimizer = optimizer
        self.edge_attr = edge_attr
        self.acc_weight = acc_weight
        self.force_weight = force_weight
        self.neighbor_weight = neighbor_weight
        self.energy_weight = energy_weight
        self.device = device

    def train_once(self, type='disk', num_scenes=10, batch_size=64, epochs=4, previous_pos=2):
        if self.energy_weight > 0:
            dataset = NBodyDataset(type=type, num_scenes=num_scenes, device=self.device, previous_pos=previous_pos, energy=True)
        else:
            dataset = NBodyDataset(type=type, num_scenes=num_scenes, device=self.device, previous_pos=previous_pos, energy=False)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        train_mse_losses = []

        for epoch in range(epochs):
            epoch_loss = []
            epoch_mse_loss = []
            for pos, feat, acc, U, K in dataloader:
                loss, mse_loss = self.model.train_batch(self.optimizer, pos, feat, acc, U, K, edge_attr=self.edge_attr, acc_weigth=self.acc_weight, force_weigth=self.force_weight, neighbor_weigth=self.neighbor_weight, energy_weigth=self.energy_weight)
                epoch_loss.append(loss)
                epoch_mse_loss.append(mse_loss)

            train_losses.append(sum(epoch_loss) / len(epoch_loss))
            train_mse_losses.append(sum(epoch_mse_loss) / len(epoch_mse_loss))

        return train_losses, train_mse_losses
    
    def train(self, runs, type='disk', num_scenes=10, batch_size=64, epochs=4, previous_pos=2, save_every=10):
        # create a folder for saving
        if save_every > 0:
            path = './models' + datetime.now().strftime("%Y%m%d%H%M%S")
            os.mkdir(path)

        for i in tqdm.tqdm(range(runs)):
            train_losses, train_mse_losses = self.train_once(type=type, num_scenes=num_scenes, batch_size=batch_size, epochs=epochs, previous_pos=previous_pos)
            print(f"Run {i+1}: Loss: {train_losses[-1]}, MSE Loss: {train_mse_losses[-1]}")
            if i % save_every == 0:
                torch.save(self.model.state_dict(), f'{path}/model_{i}.pt')
        if save_every > 0:
            torch.save(self.model.state_dict(), f'{path}/model_final.pt')




            
