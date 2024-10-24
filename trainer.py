import torch
import tqdm
from datagen import generate_dataset
import os


class Trainer:

    def __init__(self,loss_fn, batch_size, device='cuda', scheduler=None, mode='past'):
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.mode = mode

    def get_new_pos_vel(self, acc, pos, vel, dt=0.01):
        new_vel = vel + acc * dt 
        new_pos = pos + new_vel * dt
        return new_pos, new_vel


    def train_round(self, data, model, optimizer):

        self.model.train()

        num_batches = len(data) // self.batch_size
        all_losses = []
        all_dists = []
        if self.mode != 'past':
            for i in tqdm.tqdm(range(num_batches)):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                m = torch.tensor([b['masses'] for b in batch], dtype=torch.float32).to(self.device)
                pos0 = torch.tensor([b['pos'] for b in batch], dtype=torch.float32).to(self.device)
                vel0 = torch.tensor([b['vel'] for b in batch], dtype=torch.float32).to(self.device)
                acc0 = torch.tensor([b['acc'] for b in batch], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                losses = []
                for j in range(len(batch)):
                    l = 0
                    sample_masses = m[j].unsqueeze(1)
                    sample_pos0 = pos0[j]
                    sample_vel0 = vel0[j]
                    sample_acc0 = acc0[j]


                    pr_acc0 = model(sample_pos0, sample_vel0, sample_masses)
                    l+= self.loss_fn(sample_acc0, pr_acc0)


                    losses.append(l)

                total_loss = 64 * sum(losses) / len(batch)
                all_losses.append(total_loss.item())
                total_loss.backward()

                optimizer.step()
        else:
            for i in tqdm.tqdm(range(num_batches)):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                m = torch.tensor([b['masses'] for b in batch], dtype=torch.float32).to(self.device)
                pos0 = torch.tensor([b['pos'] for b in batch], dtype=torch.float32).to(self.device)
                vel0 = torch.tensor([b['vel'] for b in batch], dtype=torch.float32).to(self.device)
                acc0 = torch.tensor([b['acc'] for b in batch], dtype=torch.float32).to(self.device)
                past_pos = torch.tensor([b['past_pos'] for b in batch], dtype=torch.float32).to(self.device)


                optimizer.zero_grad()
                losses = []
                for j in range(len(batch)):
                    l = 0
                    sample_masses = m[j].unsqueeze(1)
                    sample_pos0 = pos0[j]
                    sample_vel0 = vel0[j]
                    sample_acc0 = acc0[j]
                    sample_past_pos = past_pos[j]

                    pr_acc0 = model(sample_pos0, sample_vel0, sample_masses, sample_past_pos)

                    l+= self.loss_fn(sample_acc0, pr_acc0)

                    losses.append(l)

                total_loss = 64 * sum(losses) / len(batch)
                all_losses.append(total_loss.item())
                total_loss.backward()

                optimizer.step()

        print(f'Train Loss: {sum(all_losses)/len(all_losses)}, Train L2: {sum(all_dists)/len(all_dists)}')

    def train(self, model, optimizer, rounds, epochs_per_dataset, scenes_per_dataset=25, weights_dir='./models/'):

        model.to(self.device)
        if weights_dir is not None:
            weight_paths = os.listdir(weights_dir)
            weight_paths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            try:
                model.load_state_dict(torch.load(os.path.join(weights_dir, weight_paths[-1])))
                print(f'Loaded weights from {weight_paths[-1]}')
                last_model = int(weight_paths[-1].split('_')[1].split('.')[0])
                last_model += 1
            except:
                last_model = 0

        for i in range(rounds):
            print(f'Round {i}')
            data = generate_dataset(scenes_per_dataset)
            for j in range(epochs_per_dataset):
                print(f'Epoch {j}')
                self.train_round(data, model, optimizer)
                if self.scheduler:
                    self.scheduler.step()

            if weights_dir is not None:
                torch.save(model.state_dict(), os.path.join(weights_dir, f'model_{last_model}.pt'))
                print(f'Saved weights to {os.path.join(weights_dir, f"model_{last_model}.pt")}')
                last_model += 1

        return model

        