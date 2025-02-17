import tqdm
from generation import NBodyDataset
from torch.utils.data import DataLoader
import torch
import os
from datetime import datetime

class Trainer:
    def __init__(self, model, optimizer, edge_attr, acc_weight, force_weight, neighbor_weight, energy_weight, device, dt=0.01):
        self.model = model
        self.optimizer = optimizer
        self.edge_attr = edge_attr
        self.acc_weight = acc_weight
        self.force_weight = force_weight
        self.neighbor_weight = neighbor_weight
        self.energy_weight = energy_weight
        self.device = device
        self.dt = dt

    def train_once(self, type='disk', num_scenes=10, batch_size=64, epochs=4, previous_pos=2):
        dataset = NBodyDataset(type=type, num_scenes=num_scenes, device=self.device, previous_pos=previous_pos, energy=self.energy_weight > 0, dt=self.dt)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        train_mse_losses = []

        for epoch in tqdm.tqdm(range(epochs), leave=False):
            epoch_loss = []
            epoch_mse_loss = []
            for pos, feat, acc, U, K in dataloader:
                pos, feat, acc = pos.to(self.device), feat.to(self.device), acc.to(self.device)
                
                if U.sum() == 0:
                    U, K = None, None
                else:
                    U, K = U.to(self.device), K.to(self.device)

                loss, mse_loss = self.model.train_batch(
                    self.optimizer, pos, feat, acc, U, K,
                    edge_attr=self.edge_attr, acc_weigth=self.acc_weight, 
                    force_weigth=self.force_weight, neighbor_weigth=self.neighbor_weight, 
                    energy_weigth=self.energy_weight
                )
                
                epoch_loss.append(loss)
                epoch_mse_loss.append(mse_loss)


            train_losses.append(sum(epoch_loss) / len(epoch_loss))
            train_mse_losses.append(sum(epoch_mse_loss) / len(epoch_mse_loss))

        

        return train_losses, train_mse_losses
    
    def train(self, runs, type='disk', num_scenes=10, batch_size=64, epochs=4, previous_pos=2, save_every=10, model_path=None):
        # Create a folder for saving
        if save_every > 0:
            if model_path:
                path = model_path
            else:
                path = './models' + datetime.now().strftime("%Y%m%d%H%M%S")
                os.mkdir(path)
                
        last_model = 0
        if model_path:
            models = sorted(os.listdir(model_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            with torch.no_grad():  # Prevents gradient tracking
                self.model.load_state_dict(torch.load(f'{model_path}/{models[-1]}', map_location=self.device))
            print(f"Loaded model {models[-1]}")
            last_model = int(models[-1].split('_')[1].split('.')[0])

        runs_range = tqdm.trange(runs)
        for i in runs_range:
            train_losses, train_mse_losses = self.train_once(type=type, num_scenes=num_scenes, batch_size=batch_size, epochs=epochs, previous_pos=previous_pos)
            torch.cuda.empty_cache()
            runs_range.set_postfix_str(f"Run {i+1}: Loss: {train_losses[-1]}, Euclidean Distance: {train_mse_losses[-1]}")
            
            if save_every > 0 and i % save_every == 0:
                torch.save(self.model.state_dict(), f'{path}/model_{i+last_model}.pt')

        if save_every > 0:
            torch.save(self.model.state_dict(), f'{path}/model_final.pt')
