import numpy as np
import matplotlib.pyplot as plt
# use matplotlib animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# and HTML to display the animation
from IPython.display import HTML

class Particle:
    def __init__(self, mass, position, velocity, type=None):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.zeros_like(self.position)
        self.type = type

class NBodySimulation:
    def __init__(self, particles, G=1.0, softening=0.1, dt=0.01, calc_energy=False):
        self.particles = particles
        self.G = G
        self.softening = softening
        self.dt = dt
        self.t = 0
        self.calc_energy = calc_energy
    
    def get_accelerations(self):
        positions = np.array([p.position for p in self.particles])
        masses = np.array([p.mass for p in self.particles]).reshape(-1, 1)
        
        x = positions[:, 0:1]
        y = positions[:, 1:2]
        z = positions[:, 2:3]
        
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z
        
        inv_r3 = (dx**2 + dy**2 + dz**2 + self.softening**2)
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)
        
        ax = self.G * (dx * inv_r3) @ masses
        ay = self.G * (dy * inv_r3) @ masses
        az = self.G * (dz * inv_r3) @ masses
        
        accelerations = np.hstack((ax, ay, az))
        for i, p in enumerate(self.particles):
            p.acceleration = accelerations[i]

    def get_energy(self):
        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])
        masses = np.array([p.mass for p in self.particles])
        
        KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        
        x = positions[:, 0:1]
        y = positions[:, 1:2]
        z = positions[:, 2:3]
        
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z
        
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]
        
        PE = self.G * np.sum(np.triu(-(masses * masses.T) * inv_r, 1))
        
        return KE, PE

    def run(self, t_end, save_states=False):
        num_steps = int(np.ceil(t_end / self.dt))
        
        pos_save = np.zeros((len(self.particles), 3, num_steps + 1))
        vel_save = np.zeros((len(self.particles), 3, num_steps + 1))
        accel_save = np.zeros((len(self.particles), 3, num_steps + 1))
        KE_save = np.zeros(num_steps + 1)
        PE_save = np.zeros(num_steps + 1)
        t_all = np.arange(num_steps + 1) * self.dt
        
        for i, p in enumerate(self.particles):
            pos_save[i, :, 0] = p.position
            vel_save[i, :, 0] = p.velocity
            accel_save[i, :, 0] = p.acceleration
        
        self.get_accelerations()
        if self.calc_energy:
            KE, PE = self.get_energy()
            KE_save[0] = KE
            PE_save[0] = PE

        
        for i in range(num_steps):
            for p in self.particles:
                p.velocity += p.acceleration * self.dt / 2.0
                p.position += p.velocity * self.dt
            
            self.get_accelerations()
            
            for p in self.particles:
                p.velocity += p.acceleration * self.dt / 2.0

            self.t += self.dt
            
            if self.calc_energy:
                KE, PE = self.get_energy()
                KE_save[i + 1] = KE
                PE_save[i + 1] = PE
            
            for j, p in enumerate(self.particles):
                pos_save[j, :, i + 1] = p.position
                vel_save[j, :, i + 1] = p.velocity
                accel_save[j, :, i + 1] = p.acceleration

        types = [p.type for p in self.particles]
            
        if save_states:
            return pos_save, vel_save, accel_save, KE_save, PE_save, t_all, np.array([p.mass for p in self.particles]), types

        

def generate_disk_galaxy(N, M, m, R, G, displacement, clockwise=True):
    # generate random positions in a disk, with a black hole at the center, and add angular momentum to each particle of the disk
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    masses = np.zeros(N)
    types = np.zeros(N)
    types[0] = 'black hole'
    # black hole at the center
    masses[0] = M
    positions[0] = [0, 0, 0]
    velocities[0] = [0, 0, 0]

    # disk particles
    for i in range(1, N):
        types[i] = 'star'
        r = R * np.sqrt(np.random.rand()) + 0.1
        theta = 2 * np.pi * np.random.rand()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * np.random.randn() * 0.1
        positions[i] = [x, y, z]

        if clockwise:
            v = np.sqrt(G * M / r)
            vx = v * np.sin(theta)
            vy = -v * np.cos(theta)
            vz = 0
        else:
            v = np.sqrt(G * M / r)
            vx = -v * np.sin(theta)
            vy = v * np.cos(theta)
            vz = 0
        velocities[i] = [vx, vy, vz]

        masses[i] = m

    # displacement
    for i in range(N):
        positions[i] += displacement

    
    print(types[0])

    particles = [Particle(mass, pos, vel, type=t) for mass, pos, vel, t in zip(masses, positions, velocities, types)]

    return particles

def generateDisk3D(nbStars, radius, mass, Mass, zOffsetMax, gravityCst, seed=None, offset=[0, 0, 0], initial_vel=[0,0,0], clockwise=True):
    np.random.seed(seed)

    # Calculating positions
    positions = np.zeros(shape=(nbStars + 1, 3))
    distances = np.sqrt(np.random.random((nbStars+1,))) + 0.1 * radius
    distances[0] = 0
    zOffsets = (np.random.random((nbStars+1,)) - 0.5) * 2 * zOffsetMax * (np.ones_like(distances) - np.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = np.random.random((nbStars+1,)) * 2 * np.pi
    positions[:, 0] = np.cos(angles) * distances
    positions[:, 1] = np.sin(angles) * distances
    positions[:, 2] = zOffsets

    # Calculating speeds
    velocities = np.zeros(shape=(nbStars+1, 3))
    masses = np.ones(nbStars + 1)
    masses[0] = Mass
    
    masses[1:] = masses[1:] * mass / (nbStars)
    for i in range(1, nbStars):
        mask = distances < distances[i]
        internalMass = np.sum(masses[mask])
        velNorm = np.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * np.cos(angles[i] + np.pi / 2)
        velocities[i, 1] = velNorm * np.sin(angles[i] + np.pi / 2)
        velocities[i, 2] = np.zeros_like(velocities[i, 2])

    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]


    # Adding offset
    for i in range(nbStars + 1):
        positions[i] += offset
        velocities[i] += initial_vel
    return [Particle(mass, pos, vel) for mass, pos, vel in zip(masses, positions, velocities)]

def uniformSphereDistribution(r, r0=1, M=1):
    exteriorMask = r > r0
    density = 3 * M / (4 * np.pi * r0 ** 3)
    density[exteriorMask] = 0
    return density


def isothermalSphereDistribution(r, p0=1, r0=1):
    density = p0 * (r / r0) ** (-2)
    return density


def sphericalPlummerDistribution(r, r0=1, M=1):
    density = (3 * M) / (4 * np.pi) * (r0 ** 2) / (r0 ** 2 + r ** 2) ** (3 / 2)
    return density


def sphericalHernquistDistribution(r, r0=1, M=1):
    density =(M / (2 * np.pi)) * (r0 / (r * (r0 + r) ** 3))
    return density


def sphericalJaffeDistribution(r, r0=1, M=1):
    density = M / (4 * np.pi) * r0 / (r ** 2 * (r0 + r) ** 2)
    return density

def generateDisk3Dv2(nbStars, radius, mass, Mass, zOffsetMax, gravityCst, distribution, dist_params={}, seed=None, offset=[0, 0, 0], initial_vel=[0, 0, 0], clockwise=True, angle=[0, 0, 0], t=None):
    np.random.seed(seed)
    '''
    Generate a 3D disk of stars with a central black hole
    Parameters:
    - nbStars: number of stars in the disk
    - radius: radius of the disk
    - mass: mass of each star
    - Mass: mass of the central black hole
    - zOffsetMax: maximum offset in z direction
    - gravityCst: gravitational constant
    - distribution: distribution of the stars in the disk (plummer, hernquist, jaffe, isothermal, uniform)
    - dist_params: parameters of the distribution
    - seed: random seed
    - offset: offset of the disk
    - initial_vel: initial velocity of the disk
    - clockwise: direction of rotation of the disk
    '''

    # Calculating positions
    positions = np.zeros(shape=(nbStars + 1, 3))
    distances = np.sqrt(np.random.random((nbStars+1,))) + 0.1 * radius
    types = [None] * (nbStars + 1)  
    distances[0] = 0
    types[0] = 'black hole'
    zOffsets = (np.random.random((nbStars+1,)) - 0.5) * 2 * zOffsetMax * (np.ones_like(distances) - np.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = np.random.random((nbStars+1,)) * 2 * np.pi
    positions[:, 0] = np.cos(angles) * distances
    positions[:, 1] = np.sin(angles) * distances
    positions[:, 2] = zOffsets

    # Calculating speeds
    velocities = np.zeros(shape=(nbStars+1, 3))
    
    # Assigning masses
    masses = np.ones(nbStars + 1)
    masses[0] = Mass  # Mass of central object

    # Calculate masses based on the chosen distribution
    for i in range(1, nbStars+1):
        types[i] = 'star'
        if distribution == 'plummer':
            masses[i] = sphericalPlummerDistribution(distances[i], **dist_params)
        elif distribution == 'hernquist':
            masses[i] = sphericalHernquistDistribution(distances[i], **dist_params)
        elif distribution == 'jaffe':
            masses[i] = sphericalJaffeDistribution(distances[i], **dist_params)
        elif distribution == 'isothermal':
            masses[i] = isothermalSphereDistribution(distances[i], **dist_params)
        elif distribution == 'uniform':
            masses[i] = uniformSphereDistribution(distances[i], **dist_params)
        else:
            raise ValueError("Unsupported distribution type")

    # Normalize masses to sum to 'mass'
    masses[1:] = masses[1:] * mass / np.sum(masses[1:])

    # Compute velocities
    for i in range(1, nbStars+1):
        mask = distances < distances[i]
        internalMass = np.sum(masses[mask])  # Sum of enclosed mass
        velNorm = np.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * np.cos(angles[i] + np.pi / 2)
        velocities[i, 1] = velNorm * np.sin(angles[i] + np.pi / 2)
        velocities[i, 2] = np.zeros_like(velocities[i, 2])

    # Apply clockwise or counter-clockwise rotation
    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]

    # Rotate the disk
    angle = np.array(angle)

    Rx = np.vstack([[1, 0, 0], [0, np.cos(angle[0]), -np.sin(angle[0]),], [0, np.sin(angle[0]), np.cos(angle[0])]])
    
    Ry = np.vstack([[np.cos(angle[1]), 0, np.sin(angle[1])], [0, 1, 0], [-np.sin(angle[1]), 0, np.cos(angle[1])]])

    Rz = np.vstack([[np.cos(angle[2]), -np.sin(angle[2]), 0], [np.sin(angle[2]), np.cos(angle[2]), 0], [0, 0, 1]])
    
    positions = np.dot(Rx, positions.T).T
    positions = np.dot(Ry, positions.T).T
    velocities = np.dot(Rx, velocities.T).T
    velocities = np.dot(Ry, velocities.T).T
    positions = np.dot(Rz, positions.T).T
    velocities = np.dot(Rz, velocities.T).T

    # Adding offset
    for i in range(nbStars + 1):
        positions[i] += offset
        velocities[i] += initial_vel


    return [Particle(mass, pos, vel, type=t) for mass, pos, vel, t in zip(masses, positions, velocities, types)]

def generateDisk3Dv3(nbStars, radius, Mass, zOffsetMax, gravityCst, distribution, dist_params={}, seed=None, offset=[0, 0, 0], initial_vel=[0, 0, 0], clockwise=True, angle=[0, 0, 0], t=None):
    np.random.seed(seed)
    '''
    Generate a 3D disk of stars with a central black hole
    Parameters:
    - nbStars: number of stars in the disk
    - radius: radius of the disk
    - mass: mass of each star
    - Mass: mass of the central black hole
    - zOffsetMax: maximum offset in z direction
    - gravityCst: gravitational constant
    - distribution: distribution of the stars in the disk (plummer, hernquist, jaffe, isothermal, uniform)
    - dist_params: parameters of the distribution
    - seed: random seed
    - offset: offset of the disk
    - initial_vel: initial velocity of the disk
    - clockwise: direction of rotation of the disk
    '''

    # Calculating positions
    positions = np.zeros(shape=(nbStars + 1, 3))
    distances = np.sqrt(np.random.random((nbStars+1,)))
    # if any of the distances is 0, we set it to the machine epsilon
    distances[distances == 0] = 0.01 * radius
    types = [None] * (nbStars + 1)  
    distances[0] = 0
    zOffsets = (np.random.random((nbStars+1,)) - 0.5) * 2 * zOffsetMax * (np.ones_like(distances) - np.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = np.random.random((nbStars+1,)) * 2 * np.pi
    positions[:, 0] = np.cos(angles) * distances
    positions[:, 1] = np.sin(angles) * distances
    positions[:, 2] = zOffsets

    # Calculating speeds
    velocities = np.zeros(shape=(nbStars+1, 3))
    
    # Assigning masses
    masses = np.ones(nbStars + 1)

    # Calculate masses based on the chosen distribution
    for i in range(nbStars+1):
        if i == 0:
            types[i] = 'black hole'
            dist = np.finfo(np.float32).eps
            if distribution == 'plummer':
                masses[i] = sphericalPlummerDistribution(dist, **dist_params)
            elif distribution == 'hernquist':
                masses[i] = sphericalHernquistDistribution(dist, M=Mass,r0=1)
            elif distribution == 'jaffe':
                masses[i] = sphericalJaffeDistribution(dist, **dist_params)
            elif distribution == 'isothermal':
                masses[i] = isothermalSphereDistribution(dist, **dist_params)
            elif distribution == 'uniform':
                masses[i] = uniformSphereDistribution(dist, **dist_params)
            else:
                raise ValueError("Unsupported distribution type")
        else:
            types[i] = 'star'
            if distribution == 'plummer':
                masses[i] = sphericalPlummerDistribution(distances[i], **dist_params)
            elif distribution == 'hernquist':
                masses[i] = sphericalHernquistDistribution(distances[i], M=Mass,r0=1)
            elif distribution == 'jaffe':
                masses[i] = sphericalJaffeDistribution(distances[i], **dist_params)
            elif distribution == 'isothermal':
                masses[i] = isothermalSphereDistribution(distances[i], **dist_params)
            elif distribution == 'uniform':
                masses[i] = uniformSphereDistribution(distances[i], **dist_params)
            else:
                raise ValueError("Unsupported distribution type")
            

    # Normalize masses to sum to 'Mass'
    masses = masses * Mass / np.sum(masses)



    # Compute velocities
    for i in range(1, nbStars+1):
        mask = distances < distances[i]
        internalMass = np.sum(masses[mask])  # Sum of enclosed mass
        velNorm = np.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * np.cos(angles[i] + np.pi / 2)
        velocities[i, 1] = velNorm * np.sin(angles[i] + np.pi / 2)
        velocities[i, 2] = np.zeros_like(velocities[i, 2])

    # Apply clockwise or counter-clockwise rotation
    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]

    # Rotate the disk
    angle = np.array(angle)

    Rx = np.vstack([[1, 0, 0], [0, np.cos(angle[0]), -np.sin(angle[0]),], [0, np.sin(angle[0]), np.cos(angle[0])]])
    
    Ry = np.vstack([[np.cos(angle[1]), 0, np.sin(angle[1])], [0, 1, 0], [-np.sin(angle[1]), 0, np.cos(angle[1])]])

    Rz = np.vstack([[np.cos(angle[2]), -np.sin(angle[2]), 0], [np.sin(angle[2]), np.cos(angle[2]), 0], [0, 0, 1]])
    
    positions = np.dot(Rx, positions.T).T
    positions = np.dot(Ry, positions.T).T
    velocities = np.dot(Rx, velocities.T).T
    velocities = np.dot(Ry, velocities.T).T
    positions = np.dot(Rz, positions.T).T
    velocities = np.dot(Rz, velocities.T).T

    # Adding offset
    for i in range(nbStars + 1):
        positions[i] += offset
        velocities[i] += initial_vel


    return [Particle(mass, pos, vel, type=t) for mass, pos, vel, t in zip(masses, positions, velocities, types)]

def generateDisk3Dv4(nbStars, radius, Mass, bh_Mass_percentage, zOffsetMax, gravityCst, distribution, dist_params={}, seed=None, offset=[0, 0, 0], initial_vel=[0, 0, 0], clockwise=True, angle=[0, 0, 0], t=None):
    np.random.seed(seed)
    '''
    Generate a 3D disk of stars with a central black hole
    Parameters:
    - nbStars: number of stars in the disk
    - radius: radius of the disk
    - mass: mass of each star
    - Mass: mass of the central black hole
    - zOffsetMax: maximum offset in z direction
    - gravityCst: gravitational constant
    - distribution: distribution of the stars in the disk (plummer, hernquist, jaffe, isothermal, uniform)
    - dist_params: parameters of the distribution
    - seed: random seed
    - offset: offset of the disk
    - initial_vel: initial velocity of the disk
    - clockwise: direction of rotation of the disk
    '''

    # Calculating positions
    positions = np.zeros(shape=(nbStars + 1, 3))
    distances = np.sqrt(np.random.random((nbStars+1,)))
    # if any of the distances is 0, we set it to the machine epsilon
    distances[distances == 0] = np.finfo(np.float32).eps

    types = [None] * (nbStars + 1)  
    distances[0] = 0
    zOffsets = (np.random.random((nbStars+1,)) - 0.5) * 2 * zOffsetMax * (np.ones_like(distances) - np.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = np.random.random((nbStars+1,)) * 2 * np.pi
    positions[:, 0] = np.cos(angles) * distances
    positions[:, 1] = np.sin(angles) * distances
    positions[:, 2] = zOffsets

    # Calculating speeds
    velocities = np.zeros(shape=(nbStars+1, 3))
    
    # Assigning masses
    masses = np.ones(nbStars + 1)

    masses[0] = Mass * bh_Mass_percentage
    types[0] = 'black hole'

    M = Mass * (1 - bh_Mass_percentage)

    # Calculate masses based on the chosen distribution
    for i in range(1, nbStars+1):
        types[i] = 'star'
        if distribution == 'plummer':
            masses[i] = sphericalPlummerDistribution(distances[i], **dist_params)
        elif distribution == 'hernquist':
            masses[i] = sphericalHernquistDistribution(distances[i], M=M,r0=1)
        elif distribution == 'jaffe':
            masses[i] = sphericalJaffeDistribution(distances[i], **dist_params)
        elif distribution == 'isothermal':
            masses[i] = isothermalSphereDistribution(distances[i], **dist_params)
        elif distribution == 'uniform':
            masses[i] = uniformSphereDistribution(distances[i], **dist_params)
        else:
            raise ValueError("Unsupported distribution type")
            

    # Normalize masses to sum to 'Mass'
    masses[1:] = masses[1:] * M / np.sum(masses[1:])

    # Compute velocities
    for i in range(1, nbStars+1):
        mask = distances <= distances[i]
        internalMass = np.sum(masses[mask])  # Sum of enclosed mass
        velNorm = np.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * np.cos(angles[i] + np.pi / 2)
        velocities[i, 1] = velNorm * np.sin(angles[i] + np.pi / 2)
        velocities[i, 2] = np.zeros_like(velocities[i, 2])

    # Apply clockwise or counter-clockwise rotation
    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]

    # Rotate the disk
    angle = np.array(angle)

    Rx = np.vstack([[1, 0, 0], [0, np.cos(angle[0]), -np.sin(angle[0]),], [0, np.sin(angle[0]), np.cos(angle[0])]])
    
    Ry = np.vstack([[np.cos(angle[1]), 0, np.sin(angle[1])], [0, 1, 0], [-np.sin(angle[1]), 0, np.cos(angle[1])]])

    Rz = np.vstack([[np.cos(angle[2]), -np.sin(angle[2]), 0], [np.sin(angle[2]), np.cos(angle[2]), 0], [0, 0, 1]])
    
    positions = np.dot(Rx, positions.T).T
    positions = np.dot(Ry, positions.T).T
    velocities = np.dot(Rx, velocities.T).T
    velocities = np.dot(Ry, velocities.T).T
    positions = np.dot(Rz, positions.T).T
    velocities = np.dot(Rz, velocities.T).T

    # Adding offset
    for i in range(nbStars + 1):
        positions[i] += offset
        velocities[i] += initial_vel


    return [Particle(mass, pos, vel, type=t) for mass, pos, vel, t in zip(masses, positions, velocities, types)]



def nfw_distribution(N, r_s, seed=None):
    """
    Generate N 3D positions for dark matter particles following an NFW profile.
    
    Parameters:
    - N: Number of dark matter particles
    - r_s: Scale radius for the dark matter halo
    - seed: Random seed
    
    Returns:
    - positions: A (N, 3) array of 3D positions
    """
    np.random.seed(seed)

    # Sample uniform random numbers
    u = np.random.uniform(0, 1, N)

    # Sample radial distances using the NFW distribution
    r = r_s * (u**(-1/3) - 1)  # This is an approximation to sample from the NFW profile

    # Sample random angles for spherical coordinates
    theta = np.arccos(2 * np.random.uniform(0, 1, N) - 1)  # theta: [0, pi]
    phi = np.random.uniform(0, 2 * np.pi, N)  # phi: [0, 2pi]

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.column_stack((x, y, z))
    return positions


def nfw_mass_distribution(r, r_s, total_mass):
    """
    Calculate the mass of dark matter within radius r using the NFW profile.
    
    Parameters:
    - r: Radial distance from the center (can be an array)
    - r_s: Scale radius for the NFW profile
    - total_mass: Total mass of the dark matter
    
    Returns:
    - mass: Mass corresponding to radius r
    """
    # Mass within a radius r based on the NFW profile
    M_r = r_s * (np.log(1 + r / r_s) - r / (r + r_s))
    M_r = M_r / np.max(M_r)  # Normalize to [0,1]
    M_r = M_r * total_mass    # Scale to total mass
    
    return M_r


def generateDiskWithDarkMatter(nbStars, radius, Mass, bh_Mass_percentage, zOffsetMax, gravityCst, distribution, darkMatter_percentage, dist_params={}, seed=None, offset=[0, 0, 0], initial_vel=[0, 0, 0], clockwise=True, angle=[0, 0, 0], t=None):
    """
    Generate a 3D disk of stars with a central black hole and dark matter halo.
    
    Parameters:
    - nbStars: Number of stars in the disk
    - radius: Radius of the disk
    - Mass: Mass of the central black hole
    - bh_Mass_percentage: Percentage of the mass that belongs to the black hole
    - darkMatter_percentage: Percentage of mass in dark matter halo
    - zOffsetMax: Maximum offset in z direction for stars
    - gravityCst: Gravitational constant
    - distribution: Distribution of stars in the disk (plummer, hernquist, etc.)
    - dist_params: Parameters of the distribution
    - seed: Random seed
    - offset: Offset of the disk
    - initial_vel: Initial velocity of the disk
    - clockwise: Direction of rotation of the disk
    - angle: Orientation angles for the disk
    - t: Time for evolution (not used here)
    
    Returns:
    - A list of Particle objects with positions, velocities, masses, and types.
    """
    np.random.seed(seed)

    # Stars and Black Hole
    positions = np.zeros(shape=(nbStars + 1, 3))
    distances = np.sqrt(np.random.random((nbStars+1,))) + 0.1 * radius
    distances[distances == 0] = np.finfo(np.float32).eps

    types = [None] * (nbStars + 1)
    distances[0] = 0
    zOffsets = (np.random.random((nbStars+1,)) - 0.5) * 2 * zOffsetMax * (np.ones_like(distances) - np.sqrt(distances))
    zOffsets[0] = 0
    distances = distances * radius
    angles = np.random.random((nbStars+1,)) * 2 * np.pi
    positions[:, 0] = np.cos(angles) * distances
    positions[:, 1] = np.sin(angles) * distances
    positions[:, 2] = zOffsets

    velocities = np.zeros(shape=(nbStars+1, 3))

    masses = np.ones(nbStars + 1)
    masses[0] = Mass * bh_Mass_percentage
    types[0] = 'black hole'

    # Mass distribution for stars
    M = Mass * (1 - bh_Mass_percentage) * (1 - darkMatter_percentage)
    for i in range(1, nbStars + 1):
        types[i] = 'star'
        if distribution == 'hernquist':
            masses[i] = sphericalHernquistDistribution(distances[i], M=M, r0=1)
        # other distributions can be added similarly

    masses[1:] = masses[1:] * M / np.sum(masses[1:])

    for i in range(1, nbStars + 1):
        mask = distances < distances[i]
        internalMass = np.sum(masses[mask])
        velNorm = np.sqrt(gravityCst * internalMass / distances[i])
        velocities[i, 0] = velNorm * np.cos(angles[i] + np.pi / 2)
        velocities[i, 1] = velNorm * np.sin(angles[i] + np.pi / 2)

    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]

    # Dark Matter
    nbDarkMatter = int(darkMatter_percentage * nbStars)
    r_s = radius * 2  # Halo typically more extended than stars
    darkMatter_positions = nfw_distribution(nbDarkMatter, r_s, seed=seed)
    
    # Sample dark matter masses
    darkMatter_distances = np.linalg.norm(darkMatter_positions, axis=1)
    darkMatter_masses = nfw_mass_distribution(darkMatter_distances, r_s, Mass * darkMatter_percentage)

    # Concatenate star and dark matter positions and masses
    total_positions = np.vstack((positions, darkMatter_positions))
    total_masses = np.concatenate((masses, darkMatter_masses))
    total_types = types + ['dark matter'] * nbDarkMatter

    return [Particle(mass, pos, vel, type=t) for mass, pos, vel, t in zip(total_masses, total_positions, np.vstack((velocities, np.zeros((nbDarkMatter, 3)))), total_types)]


