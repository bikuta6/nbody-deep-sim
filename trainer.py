import torch
import numpy as np
import tqdm
from datagen import generate_dataset, generate_dataset_past, generate_dataset_transformer
import os
import gc


class Trainer:

    def __init__(self,loss_fn, batch_size, device='cuda', mode='past'):
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.mode = mode

    def get_new_pos_vel(self, acc, pos, vel, dt=0.01):
        new_vel = vel + acc * dt 
        new_pos = pos + new_vel * dt
        return new_pos, new_vel


    def train_round(self, data, model, optimizer):
        num_batches = len(data) // self.batch_size
        all_losses = []
        if self.mode == 'present':
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

                    feats = torch.cat([sample_masses, sample_vel0], dim=1)


                    pr_acc0 = model(feats, sample_pos0)
                    l+= self.loss_fn(sample_acc0, pr_acc0)


                    losses.append(l)
                
                del m
                del pos0
                del vel0
                del acc0

                torch.cuda.empty_cache()

                total_loss = sum(losses) / len(batch)
                all_losses.append(total_loss.item())
                total_loss.backward()

                optimizer.step()

        elif self.mode == 'transformer':
            for i in tqdm.tqdm(range(num_batches)):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                inputs = torch.tensor(np.array([b['inputs'] for b in batch]), dtype=torch.float32).to(self.device)
                accelerations = torch.tensor(np.array([b['accelerations'] for b in batch]), dtype=torch.float32).to(self.device)


                optimizer.zero_grad()
                losses = []
                for j in range(len(batch)):
                    l = 0
                    sample_inputs = inputs[j]
                    sample_acc = accelerations[j]

                    pr_acc = model(sample_inputs)

                    l+= self.loss_fn(sample_acc, pr_acc)

                    losses.append(l)
                total_loss = sum(losses) / len(batch)
                all_losses.append(total_loss.item())
                total_loss.backward()

                optimizer.step()
                del inputs
                del accelerations
                torch.cuda.empty_cache()
        elif self.mode == 'past':
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
                total_loss = sum(losses) / len(batch)
                all_losses.append(total_loss.item())
                total_loss.backward()

                optimizer.step()
                del m
                del pos0
                del vel0
                del acc0
                del past_pos
                torch.cuda.empty_cache()

        print(f'Train Loss: {sum(all_losses)/len(all_losses)}')

    def train(self, model, optimizer, scheduler, rounds, epochs_per_dataset, save_after=5,scenes_per_dataset=25, weights_dir='./models/'):
        model.train()
        model.to(self.device)
        if weights_dir is not None:
            weight_paths = os.listdir(weights_dir)
            try:
                weight_paths.remove('.ipynb_checkpoints')
            except:
                pass
            weight_paths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            try:
                model.load_state_dict(torch.load(os.path.join(weights_dir, weight_paths[-1])))
                print(f'Loaded weights from {weight_paths[-1]}')
                last_model = int(weight_paths[-1].split('_')[1].split('.')[0])
                last_model += 1
            except:
                print("No saved weights, starting from zero...")
                last_model = 0

        for i in range(rounds):
            print(f'Round {i}')
            if self.mode == 'past':
                data = generate_dataset_past(scenes_per_dataset)
            elif self.mode == 'transformer':
                data = generate_dataset_transformer(scenes_per_dataset)
            else:
                data = generate_dataset(scenes_per_dataset)
            for j in range(epochs_per_dataset):
                print(f'Epoch {j}')
                self.train_round(data, model, optimizer)
                if scheduler:
                    scheduler.step()

            if (weights_dir is not None) and (i % save_after == 0):
                torch.save(model.state_dict(), os.path.join(weights_dir, f'model_{last_model}.pt'))
                print(f'Saved weights to {os.path.join(weights_dir, f"model_{last_model}.pt")}')
                last_model += 1
            del data
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        return model

# !nvidia-smi


