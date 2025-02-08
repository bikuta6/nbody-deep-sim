import torch

class Simulator:
    def __init__(self, positions, velocities, masses, G=1.0, softening=0.1, dt=0.01, energy=True, device='cpu'):
        self.positions = positions.to(device)
        self.velocities = velocities.to(device)
        self.masses = masses.to(device)
        self.G = G
        self.softening = softening
        self.dt = dt
        self.device = device
        self.acc = torch.zeros_like(self.positions)
        self.memory = []
        self.calc_energy = energy

    def accelerations(self):
        x, y, z = self.positions[:, 0:1], self.positions[:, 1:2], self.positions[:, 2:3]
        dx = x.T - x  # Δx = x_j - x_i
        dy = y.T - y  # Δy = y_j - y_i
        dz = z.T - z  # Δz = z_j - z_i
        # Compute inverse cube of the distance with softening
        if self.softening == 0:
            inv_r3 = (dx**2 + dy**2 + dz**2)
        else:
            inv_r3 = dx**2 + dy**2 + dz**2 + self.softening**2  # Compute r_ij^2
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

        # Compute accelerations for each particle
        masses = self.masses.reshape(-1, 1)
        ax = self.G * (dx * inv_r3) @ masses
        ay = self.G * (dy * inv_r3) @ masses
        az = self.G * (dz * inv_r3) @ masses

        self.acc = torch.cat([ax, ay, az], dim=1)
    
    def energy(self):
        x, y, z = self.positions[:, 0:1], self.positions[:, 1:2], self.positions[:, 2:3]
        dx = x.T - x  # Pairwise difference in x-coordinates
        dy = y.T - y  # Pairwise difference in y-coordinates
        dz = z.T - z  # Pairwise difference in z-coordinates

        # Compute pairwise squared distances with softening
        if self.softening == 0:
            r2 = dx**2 + dy**2 + dz**2
        else:
            r2 = dx**2 + dy**2 + dz**2 + self.softening**2  # Apply softening
        inv_r = torch.where(r2 > 0, r2**-0.5, torch.tensor(0.0, device=self.device))  # 1/r_ij

        # Compute potential energy (U) for pairwise interactions
        # We compute the potential energy between particles i and j
        U = -0.5 * self.G * torch.sum(self.masses[:, None] * self.masses * inv_r)  # Double sum over i, j

        # Compute kinetic energy (K)
        v2 = torch.sum(self.velocities**2, dim=1)  # Sum of squared velocities
        K = 0.5 * torch.sum(self.masses * v2)  # Kinetic energy

        return U, K
    
    def reset_memory(self):
        self.memory = []

    def step(self):
        # Update velocities: v(t+dt) = v(t) + a(t)*dt / 2 to make the leapfrog integrator time-symmetric
        self.velocities += self.acc * self.dt / 2.0

        # Update positions: r(t+dt) = r(t) + v(t)*dt
        self.positions += self.velocities * self.dt

        # Update accelerations based on the new positions
        self.accelerations()  # Recompute accelerations with new positions

        if self.calc_energy:
            U, K = self.energy()
            self.memory[-1]['U'] = U    
            self.memory[-1]['K'] = K

            self.memory.append({
                'positions': self.positions.clone(),
                'velocities': self.velocities.clone(),
                'accelerations': self.acc.clone(),
            })

        else:
            self.memory.append({
                'positions': self.positions.clone(),
                'velocities': self.velocities.clone(),
                'accelerations': self.acc.clone()
            })

        # Update velocities: v(t+dt) = v(t) + a(t)*dt / 2 to complete the leapfrog step
        self.velocities += self.acc * self.dt / 2.0


    def run(self, steps):

        self.accelerations()

        self.memory.append({
            'positions': self.positions.clone(),
            'velocities': self.velocities.clone(),
            'accelerations': self.acc.clone()
        })

        for _ in range(steps):
            self.step()

        if self.calc_energy:
            U, K = self.energy()
            self.memory[-1]['U'] = U
            self.memory[-1]['K'] = K

        return self.memory
    

def sphericalHernquistDistribution(r, r0=1, M=1):
    density =(M / (2 * torch.pi)) * (r0 / (r * (r0 + r) ** 3))
    return density



def generateDisk3Dv3(nbStars, radius, Mass, zOffsetMax, gravityCst, seed=None, offset=[0, 0, 0], initial_vel=[0, 0, 0], clockwise=True, angle=[0, 0, 0], t=None, device='cpu'):
    if seed:
        torch.manual_seed(seed)
    
    positions = torch.zeros((nbStars + 1, 3), device=device)
    distances = torch.sqrt(torch.rand((nbStars+1,), device=device))
    distances[distances == 0] = 0.01 * radius
    types = [None] * (nbStars + 1)  
    distances[0] = 0
    zOffsets = (torch.rand((nbStars+1,), device=device) - 0.5) * 2 * zOffsetMax * (1 - torch.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = torch.rand((nbStars+1,), device=device) * 2 * torch.pi
    
    positions[:, 0] = torch.cos(angles) * distances
    positions[:, 1] = torch.sin(angles) * distances
    positions[:, 2] = zOffsets
    
    velocities = torch.zeros((nbStars+1, 3), device=device)
    masses = torch.ones(nbStars + 1, device=device)
    
    for i in range(nbStars+1):
        if i == 0:
            types[i] = 'black hole'
            dist = torch.finfo(torch.float32).eps
        else:
            types[i] = 'star'
            dist = distances[i]
        

        masses[i] = sphericalHernquistDistribution(dist, M=Mass, r0=1)

    
    masses = masses * Mass / masses.sum()
    
    for i in range(1, nbStars+1):
        mask = distances < distances[i]
        internalMass = masses[mask].sum()
        velNorm = torch.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * torch.cos(angles[i] + torch.pi / 2)
        velocities[i, 1] = velNorm * torch.sin(angles[i] + torch.pi / 2)
        velocities[i, 2] = 0.0
    
    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]
    
    angle = torch.tensor(angle, device=device)
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(angle[0]), -torch.sin(angle[0])], [0, torch.sin(angle[0]), torch.cos(angle[0])]], device=device)
    Ry = torch.tensor([[torch.cos(angle[1]), 0, torch.sin(angle[1])], [0, 1, 0], [-torch.sin(angle[1]), 0, torch.cos(angle[1])]], device=device)
    Rz = torch.tensor([[torch.cos(angle[2]), -torch.sin(angle[2]), 0], [torch.sin(angle[2]), torch.cos(angle[2]), 0], [0, 0, 1]], device=device)
    
    positions = positions @ Rx.T @ Ry.T @ Rz.T
    velocities = velocities @ Rx.T @ Ry.T @ Rz.T
    
    positions += torch.tensor(offset, device=device)
    velocities += torch.tensor(initial_vel, device=device)
    
    return positions, velocities, masses

