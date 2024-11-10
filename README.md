# nbody-deep-sim

## Objective

The aim of this project is to:

1. Test if n-body simulations can be performed using deep learning techniques.
2. Compare the performance of deep learning based n-body simulations with traditional n-body simulations.

The inspiration for this project comes from the multiple implementations of deep learning based fluid simulations, which, although not being as accurate as traditional fluid simulations, are much faster and can be used in real-time applications or for academic purposes, as seen in the paper [B. Ummenhofer and V. Koltun, Lagrangian Fluid Simulation with Continuous Convolutions, ICLR 2020](https://ge.in.tum.de/publications/2020-ummenhofer-iclr/) or [Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, Peter W. Battaglia, Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405). 

Given these implementation only conver fluid simulations, the aim of this project is to test if similar techniques can be used for n-body simulations.

## Methodology

The data used for training the deep learning models is generated using a traditional n-body simulation, described in the `sim.py` file. The data is generated for a system of `n` particles, each with a mass, position and velocity. The data is generated for `t` time steps, where the position and velocity of each particle is recorded at each time step. It uses simple matrix operation to calculate the force acting on each particle at each time step as well as the potentian and kinetic energy of the system. The benefit of this approach is that the data is generated for a system of any number of particles at any initial state and any number of time steps, leading to infinite training data.

The deep learning architectures used for this task were:

1. Continuous Convolutions

2. Graph Neural Networks

Keeping in mind that these architectures have been adapted for the task at hand, which in this case will be to predict the acceleration of each particle at each time step given data of the current state plus (optionally) one or more previous states.

### Features

The features introduced to the model are the position, velocity and mass of each particle at each time step adding the gravitational constant as well as a softening factor to avoid numerical instability. In case of taking multiple previous states as input, the features would be the concatenation of the features of the current state and the previous positions.

### Loss Function

The loss function used for this task is the mean euclidean distance between the predicted acceleration and the actual acceleration of each particle at each time step in all three dimensions. The loss is calculated as follows:

```python
def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=-1) + 1e-12)

def mean_distance(a, b):
    return torch.mean(euclidean_distance(a, b))
```

Other loss functions based on this one that were used are scaling the loss by the mass of the particle (normalized to avoid numerical instability) or by the normalized exponential of the number of neighbors of the particle, but they did not show any improvement in the results in the case of the continuous convolutions.

### Continuous Convolutions

As said before, Continuous convolutions are able to perforn similar tasks as traditional convolutional neural networks, in this case being able to take continous data as input. In this case the input would be the different particles with their respective positions, velocities and masses. The output would be the acceleration of each particle at that time step. To summarize, continuous convolutions create an sphere around each particle, creating a voxel grid (optionally using a mapping function to map the spehere into a cube) and then applying three-dimensional convolutions to the grid. These filters created by the convolutional layers are the new features that will be used to predict the acceleration of each particle.

Although at first glance this architecture seems to be the most suitable for the task at hand, it has some limitations. The main issue with this architecture, that could explain why it worked with fluids but not with n-body simulations, is that these filters do not contain information about partcles that are far away from each other. This is a problem because in n-body simulations, the force acting on a particle is the sum of the forces acting on it from all other particles, which means that the acceleration of a particle is dependent on the position of all other particles in the system. This is not the case with fluids, where the force acting on a particle is only dependent on the particles that are close to it. This is specially bad for teh case of simulating bodies in orbit of a massive body, where the force acting on a particle is extremely dependent on the position of the massive body. 

Multiple layers of continuous convolutions were used, as multiple passes might 'pass' the information of far away particles to the filters of the next layer, but experiments showed that this was not the case. Maybe with more layers or higher resolution of the grid this could be achieved, but this would also increase the computational cost of the model, defeating the purpose of using deep learning for n-body simulations.

Multiple architectures involving continuous convolutions were tested, such as:

- A simple continuous convolutional neural network with fully connected layers working in parallel with the filters, adding the features of the filters to the output of the fully connected layers.

- Using the architecture with multiple previous states as input, to see if the model could learn the dynamics of the system.

- Using attention mechanisms as well as multiple time steps of the simulation as input.

All of them failed to learn the dynamics of the system, showing that continuous convolutions are not suitable for n-body simulations.

### Graph Neural Networks

Work in progress...


## Simulator

Two main classes:

**Particle**: This class represents an individual particle with properties including mass, position, velocity, and an optional type attribute. Each particle’s initial acceleration is set to zero.

**NBodySimulation**: This class handles the simulation, computing gravitational interactions among particles and updating their positions, velocities, and accelerations over time. Key parameters include:

- G: Gravitational constant (default is 1.0).
- softening: A factor to avoid singularities when particles are too close (0.1 used for this project).
- dt: Time step for the integration.
- calc_energy: If True, the simulation calculates and tracks the kinetic and potential energy of the system at each step.

The main methods include:

- get_accelerations(): Computes the gravitational acceleration acting on each particle due to all other particles, using a softening parameter to stabilize close interactions.
- get_energy(): Returns the system's kinetic and potential energy, if calc_energy is enabled.
- run(t_end, save_states=False): Advances the simulation by a total time t_end, updating each particle’s position and velocity based on gravitational accelerations and the chosen time step dt. If save_states is set to True, it records positions, velocities, and accelerations at each step, along with optional energy values if calc_energy is enabled.

Some functions are also provided to create galaxy-like initial conditions, as well as to plot the system's evolution and energy conservation.

