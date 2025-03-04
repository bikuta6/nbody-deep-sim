import tqdm
from datautils import get_dataloader
import torch
import os
from datetime import datetime
from glob import glob
import pandas as pd
import time

class Trainer:
    def __init__(self, model, optimizer,device, dt=0.01):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dt = dt
        self.model = self.model.to(self.device)

    def train_from_dir(self, data_path, epochs, batch_size, save_every, save_path=None, create_save_path=False):

        if save_every > 0:
            if save_path:
                path = save_path
            else:
                if create_save_path:
                    path = './models' + datetime.now().strftime("%Y%m%d%H%M%S")
                    os.mkdir(path)
                
        last_model = 0
        if save_path:
            models = sorted(os.listdir(save_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            try:
                with torch.no_grad():  # Prevents gradient tracking
                    self.model.load_state_dict(torch.load(f'{save_path}/{models[-1]}', map_location=self.device))
                print(f"Loaded model {models[-1]}")
            except:
                print("No model found")

        epochs_range = tqdm.trange(epochs)
        csv_files = glob(data_path + '/*.csv')
        csv_files = [f.replace('\\', '/') for f in csv_files]

        loaders = [get_dataloader(csv_path=f, batch_size=batch_size, k=self.model.neighbors) for f in csv_files]
        
        for epoch in epochs_range:
            epoch_loss = []
            epoch_mse_loss = []
            for loader in loaders:
                for data in loader:
                    data = data.to(self.device)
                    loss, mse_loss = self.model.train_graph_batch(self.optimizer, data)
                    epoch_loss.append(loss)
                    epoch_mse_loss.append(mse_loss)

            epochs_range.set_postfix_str(f"Epoch {epoch+1}: Loss: {sum(epoch_loss) / len(epoch_loss)}, MSE: {sum(epoch_mse_loss) / len(epoch_mse_loss)}")
            
            if save_path or create_save_path:
                if (save_every > 0) and ((epoch+1) % save_every == 0):
                    torch.save(self.model.state_dict(), f'{path}/model_{epoch+1+last_model}.pt')


    def test_from_dir(self, data_path, model_path=None, sim_steps=1000, stepwise=True, rollout=True):
        if model_path:
            models = sorted(os.listdir(model_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            with torch.no_grad():  # Prevents gradient tracking
                self.model.load_state_dict(torch.load(f'{model_path}/{models[-1]}', map_location=self.device))
            print(f"Loaded model {models[-1]}")

        csv_files = glob(data_path + '/*.csv')
        csv_files = [f.replace('\\', '/') for f in csv_files]

        stepwise_loaders = [get_dataloader(csv_path=f, batch_size=1, k=self.model.neighbors, shuffle=False) for f in csv_files]
        rollout_loaders = [get_dataloader(csv_path=f, batch_size=sim_steps, k=self.model.neighbors, shuffle=False) for f in csv_files]

        df_stepwise = pd.DataFrame(columns=['filename', 'scene', 'step', 'loss', 'mse_loss', 'step_time'])
        df_rollout = pd.DataFrame(columns=['filename', 'scene', 'step', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'pred_x', 'pred_y', 'pred_z', 'pred_vx', 'pred_vy', 'pred_vz', 'pred_ax', 'pred_ay', 'pred_az', 'step_time'])

        # Stepwise evaluation
        if stepwise:
            for i, loader in tqdm.tqdm(enumerate(stepwise_loaders), total=len(stepwise_loaders), desc='Stepwise evaluation'):
                filename  = csv_files[i].split('/')[-1]
                df_stepwise = self.evaluate_stepwise(filename, loader, df_stepwise)

        # Rollout evaluation
        if rollout:
            for i, loader in tqdm.tqdm(enumerate(rollout_loaders), total=len(rollout_loaders), desc='Rollout evaluation'):
                filename  = csv_files[i].split('/')[-1]
                for scene, data in enumerate(loader):
                    data = data.to(self.device)
                    df_rollout = self.evaluate_rollout(filename, data, scene, sim_steps, self.dt, df_rollout)
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
        
        for col in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']:
            df_rollout[f'error_{col}'] = df_rollout[col] - df_rollout[f'pred_{col}']
        
        df_rollout= df_rollout.groupby(['filename', 'scene', 'step'])[[f'error_{col}' for col in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']]].mean()
        df_rollout['pos_mse'] = (df_rollout[['error_x', 'error_y', 'error_z']] ** 2).mean(axis=1)
        df_rollout['vel_mse'] = (df_rollout[['error_vx', 'error_vy', 'error_vz']] ** 2).mean(axis=1)
        df_rollout['acc_mse'] = (df_rollout[['error_ax', 'error_ay', 'error_az']] ** 2).mean(axis=1)

        return df_stepwise.groupby(['filename', 'scene']).mean()[['loss', 'step_time']], df_rollout[['pos_mse', 'vel_mse', 'acc_mse']]
    
    def evaluate_stepwise(self, filename, loader, df):
        for data in loader:
            data = data.to(self.device)
            loss, mse_loss, step_time = self.model.eval_graph_batch(data)
            new_row = {'filename': filename, 'scene': data.scene[0].item(), 'step': data.step[0].item(), 'loss': loss, 'mse_loss': mse_loss, 'step_time': step_time}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df
    
    def step(self, pos, vel, m, acc, dt):
        # Calculamos las velocidades en medio paso...
        vel_ = vel + 0.5 * dt * acc
        # ...luego las posiciones con esas velocidades...
        pos_ = pos +  dt * vel_
        # ...luego las nuevas aceleraciones en las posiciones actualizadas...
        acc_ = self.model.predict(pos_, torch.cat([vel_, m], dim=-1))
        # ...y por último actualizamos de nuevo las posiciones
        vel_ += 0.5 * dt * acc_
        return pos_, vel_, acc_
    
    def evaluate_rollout(self, filename, data, scene, sim_steps, dt, df):
        # Ensure data is on CUDA
        data = data.to(self.device)  # Move all data to the same device

        # Initial conditions
        mask = data.step == 0
        feats = data.x[mask]
        accs = data.y[mask]
        pos, vel, m = feats[:, :3], feats[:, 3:6], feats[:, 6:]

        start = time.time()
        pred_accs = self.model.predict(pos, feats[:, 3:])
        end = time.time()
        step_time = end - start

        # Store results efficiently
        rows = []

        for i in range(len(pos)):
            rows.append([
                filename, scene, 0,
                pos[i, 0].item(), pos[i, 1].item(), pos[i, 2].item(),
                vel[i, 0].item(), vel[i, 1].item(), vel[i, 2].item(),
                accs[i, 0].item(), accs[i, 1].item(), accs[i, 2].item(),
                pos[i, 0].item(), pos[i, 1].item(), pos[i, 2].item(),
                vel[i, 0].item(), vel[i, 1].item(), vel[i, 2].item(),
                pred_accs[i, 0].item(), pred_accs[i, 1].item(), pred_accs[i, 2].item(),
                step_time
            ])

        # Iteratively compute rollouts
        for step in range(1, sim_steps):
            start = time.time()
            pos, vel, pred_accs = self.step(pos, vel, m, pred_accs, dt)
            end = time.time()
            step_time = end - start

            gt_mask = data.step == step
            gt_feats = data.x[gt_mask]
            gt_accs = data.y[gt_mask]
            gt_pos, gt_vel = gt_feats[:, :3], gt_feats[:, 3:6]

            for i in range(len(pos)):
                rows.append([
                    filename, scene, step,
                    gt_pos[i, 0].item(), gt_pos[i, 1].item(), gt_pos[i, 2].item(),
                    gt_vel[i, 0].item(), gt_vel[i, 1].item(), gt_vel[i, 2].item(),
                    gt_accs[i, 0].item(), gt_accs[i, 1].item(), gt_accs[i, 2].item(),
                    pos[i, 0].item(), pos[i, 1].item(), pos[i, 2].item(),
                    vel[i, 0].item(), vel[i, 1].item(), vel[i, 2].item(),
                    pred_accs[i, 0].item(), pred_accs[i, 1].item(), pred_accs[i, 2].item(),
                    step_time
                ])

            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Convert to DataFrame in bulk to improve efficiency
        columns = [
            'filename', 'scene', 'step',
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'pred_x', 'pred_y', 'pred_z', 'pred_vx', 'pred_vy', 'pred_vz',
            'pred_ax', 'pred_ay', 'pred_az', 'step_time'
        ]
        df_new = pd.DataFrame(rows, columns=columns)

        return pd.concat([df, df_new], ignore_index=True)

        






