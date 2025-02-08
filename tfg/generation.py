import tqdm
import torch
import random
from simulator import Simulator, generateDisk3Dv3
from torch.utils.data import Dataset

def gen_params(device="cpu"):
    return {
        'nbStars': 250,
        'radius': 1,
        'Mass': 1,
        'zOffsetMax': torch.rand(1, device=device).item() * 0.5,
        'gravityCst': 1.0,
        'offset': (torch.rand(3, device=device) * 2 - 1).tolist(),
        'initial_vel': (torch.rand(3, device=device) * 0.2 - 0.1).tolist(),
        'clockwise': int(torch.randint(0, 2, (1,), device=device).item()),
        'angle': ((torch.rand(3, device=device) * 2 - 1) * 2 * torch.pi).tolist()
    }

def generate_scene_random(N, frames=1000, device='cpu', energy=False):

    pos = torch.rand((N, 3), device=device) * 6 - 3
    vel = torch.rand((N, 3), device=device) * 0.2 - 0.1
    masses = torch.rand((N,), device=device)
    masses = masses / masses.sum()

    sim = Simulator(positions=pos, velocities=vel, masses=masses, device=device, energy=energy)

    scene = sim.run(frames)

    return scene, masses

def generate_scene_disk(frames=1000, device='cpu', energy=False):
    params = gen_params(device=device)
    pos, vel, mass = generateDisk3Dv3(**params, device=device)
    
    sim = Simulator(positions=pos, velocities=vel, masses=mass, device=device, energy=energy)

    scene = sim.run(frames)

    return scene, mass

def generate_scene_multidisk(num_disks, frames=1000, device='cpu', energy=False):
    params = [gen_params(device=device) for _ in range(num_disks)]

    pos, vel, mass = generateDisk3Dv3(**params[0], device=device)

    for i in range(1, num_disks):
        pos_, vel_, mass_ = generateDisk3Dv3(**params[i], device=device)
        # stack the positions, velocities and masses
        pos = torch.cat((pos, pos_), dim=0)
        vel = torch.cat((vel, vel_), dim=0)
        mass = torch.cat((mass, mass_), dim=0)

    sim = Simulator(positions=pos, velocities=vel, masses=mass, device=device, energy=energy)

    scene = sim.run(frames)

    return scene, mass


class NBodyDataset(Dataset):
    def __init__(self, type='disk', num_disks=1, num_scenes=10, frames=1000, device='cpu', previous_pos=0, energy=False):
        self.device = device
        self.energy = energy
        self.positions = []
        self.feats = []
        self.y = []

        self.U = []
        self.K = []
        
        for _ in tqdm.tqdm(range(num_scenes)):
            if type == 'disk':
                scene, mass = generate_scene_disk(frames=frames, device=device, energy=energy)
            elif type == 'random':
                scene, mass = generate_scene_random(250, frames=frames, device=device, energy=energy)
            elif type == 'multidisk':
                scene, mass = generate_scene_multidisk(num_disks, frames=frames, device=device, energy=energy)
            else:
                raise ValueError(f"Unknown type {type}")
            
            positions = torch.stack([s['positions'] for s in scene], dim=0)
            velocities = torch.stack([s['velocities'] for s in scene], dim=0)

            if self.energy:
                Us = torch.stack([s['U'] for s in scene], dim=0)
                Ks = torch.stack([s['K'] for s in scene], dim=0)

            pos_feat_list = []
    
            for i in range(previous_pos):
                shifted_positions = torch.roll(positions, shifts=i+1, dims=0)
                shifted_positions[:i+1] = 0  # Set the first `i+1` frames to 0 (no previous data available)
                pos_feat_list.append(shifted_positions)
            
            if previous_pos > 0:
                pos_feat_list = torch.cat(pos_feat_list, dim=-1)
                feats = torch.cat((mass.unsqueeze(0).repeat(positions.size(0), 1).unsqueeze(-1), velocities, pos_feat_list), dim=-1)
            else:
                feats = torch.cat((mass.unsqueeze(0).repeat(positions.size(0), 1).unsqueeze(-1), velocities), dim=-1)

            y = torch.stack([s['accelerations'] for s in scene], dim=0) 
            self.positions.append(positions)
            self.feats.append(feats)
            self.y.append(y)

            if self.energy:
                self.U.append(Us)
                self.K.append(Ks)
        

        self.positions = torch.cat(self.positions, dim=0)
        self.feats = torch.cat(self.feats, dim=0)
        self.y = torch.cat(self.y, dim=0)
        if self.energy:
            self.U = torch.cat(self.U, dim=0)
            self.K = torch.cat(self.K, dim=0)
        
    def __len__(self):
        return self.feats.shape[0]
    
    def __getitem__(self, idx):
        if not self.energy:
            return self.positions[idx], self.feats[idx], self.y[idx], None, None
        return self.positions[idx], self.feats[idx], self.y[idx], self.U[idx], self.K[idx]
    






    
