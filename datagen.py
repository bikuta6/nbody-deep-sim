from sim import NBodySimulation, generateDisk3Dv3, Particle
import numpy as np
import tqdm
import random

def gen_params():
    return {
        'nbStars':250,
        'radius': 1,
        'Mass': 1,
        'zOffsetMax': float(np.random.uniform(0, 0.5)),
        'gravityCst': 1.0,
        'distribution': 'hernquist',
        'offset': [float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1))],
        'initial_vel': [float(np.random.uniform(-0.1, 0.1)), float(np.random.uniform(-0.1, 0.1)), float(np.random.uniform(-0.1, 0.1))],
        'clockwise': int(np.random.choice([1, 0])),
        'angle': [float(np.random.uniform(-1, 1)*2*np.pi), float(np.random.uniform(-1, 1)*2*np.pi), float(np.random.uniform(-1, 1)*2*np.pi)]
    }


def generate_scene_2gals():
    params1 = gen_params()
    params2 = gen_params()

    particles1 = generateDisk3Dv3(**params1)
    particles2 = generateDisk3Dv3(**params2)

    t_end = 10.0
    dt = 0.01
    softening = 0.1
    G = 1.0

    particles = particles1 + particles2
    sim = NBodySimulation(particles, G, softening, dt)

    pos, vel, acc, KE, PE, _, masses, types = sim.run(t_end=t_end, save_states=True)
    
    # Convert all arrays to lists
    pos = np.array(pos).transpose(2, 0, 1)
    vel = np.array(vel).transpose(2, 0, 1)
    acc = np.array(acc).transpose(2, 0, 1)
    KE = KE.flatten().astype(float).tolist()  # Ensure floats
    PE = PE.flatten().astype(float).tolist()  # Ensure floats
    masses = masses.flatten().astype(float).tolist()  # Ensure floats


    frames = []
    for i in range(len(pos)):
        frames.append({
            'frame': int(i),  # Ensure the frame index is an int
            'pos': pos[i].tolist(),
            'vel': vel[i].tolist(),
            'acc': acc[i].tolist()
        })

    final_json = {
        'galaxy1_params': params1,
        'galaxy2_params': params2,
        'dt': float(dt),
        'softening': float(softening),
        'G': float(G),
        't_end': float(t_end),
        'masses': [float(m) for m in masses],
        'types': types,
        'KE': KE,
        'PE': PE,
        'frames': frames
    }

    return final_json

def generate_scene_1gal(particles=500):
    params = {
        'nbStars': particles,
        'radius': 1,
        'Mass': 1,
        'zOffsetMax': float(np.random.uniform(0, 0.1)),
        'gravityCst': 1.0,
        'distribution': 'hernquist',
        'offset': [float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1))],
        'initial_vel': [float(np.random.uniform(-0.1, 0.1)), float(np.random.uniform(-0.1, 0.1)), float(np.random.uniform(-0.1, 0.1))],
        'clockwise': int(np.random.choice([0])),
        'angle': [float(np.random.uniform(-1, 1)*2*np.pi/4), float(np.random.uniform(-1, 1)*2*np.pi/4), float(np.random.uniform(-1, 1)*2*np.pi/4)]
    }
    particles = generateDisk3Dv3(**params)

    t_end = 10.0
    dt = 0.01
    softening = 0.1
    G = 1.0

    sim = NBodySimulation(particles, G, softening, dt)

    pos, vel, acc, KE, PE, _, masses, types = sim.run(t_end=t_end, save_states=True)

    # Convert all arrays to lists
    pos = np.array(pos).transpose(2, 0, 1)
    vel = np.array(vel).transpose(2, 0, 1)
    acc = np.array(acc).transpose(2, 0, 1)
    KE = KE.flatten().astype(float).tolist()  # Ensure floats
    PE = PE.flatten().astype(float).tolist()  # Ensure floats
    masses = masses.flatten().astype(float).tolist()  # Ensure floats

    frames = []
    for i in range(len(pos)):
        frames.append({
            'frame': int(i),  # Ensure the frame index is an int
            'pos': pos[i].tolist(),
            'vel': vel[i].tolist(),
            'acc': acc[i].tolist()
        })

    final_json = {
        'galaxy_params': params,
        'dt': float(dt),
        'softening': float(softening),
        'G': float(G),
        't_end': float(t_end),
        'masses': [float(m) for m in masses],
        'KE': KE,
        'PE': PE,
        'frames': frames,
        'types': types
    }

    return final_json

def generate_n_body_scene(n_bodies=3):
    particles = []
    for _ in range(n_bodies):
        particles.append(Particle(mass=np.random.uniform(0.4, 1.0), position=np.random.uniform(-2, 2, 3), velocity=np.random.uniform(-0.1, 0.1, 3)))

    t_end = 10.0
    dt = 0.01
    softening = 0.1
    G = 1.0

    sim = NBodySimulation(particles, G, softening, dt)

    pos, vel, acc, KE, PE, _, masses, types = sim.run(t_end=t_end, save_states=True)

    # Convert all arrays to lists
    pos = np.array(pos).transpose(2, 0, 1)
    vel = np.array(vel).transpose(2, 0, 1)
    acc = np.array(acc).transpose(2, 0, 1)
    KE = KE.flatten().astype(float).tolist()  # Ensure floats
    PE = PE.flatten().astype(float).tolist()  # Ensure floats
    masses = masses.flatten().astype(float).tolist()  # Ensure floats

    frames = []
    for i in range(len(pos)):
        frames.append({
            'frame': int(i),  # Ensure the frame index is an int
            'pos': pos[i].tolist(),
            'vel': vel[i].tolist(),
            'acc': acc[i].tolist()
        })

    final_json = {
        'dt': float(dt),
        'softening': float(softening),
        'G': float(G),
        't_end': float(t_end),
        'masses': [float(m) for m in masses],
        'KE': KE,
        'PE': PE,
        'frames': frames
    }

    return final_json



def generate_dataset_nbodies(n_scenes=5, n_bodies=3, shuffle=True):
    dataset = []
    print(f'Generating {n_bodies}-body dataset with {n_scenes} scenes...')
    for _ in tqdm.tqdm(range(n_scenes)):
        scene = generate_n_body_scene(n_bodies)
        frames = scene['frames']
        masses = scene['masses']
        for j in range(len(frames)-1):
            sample = {
                'masses': masses,
                'pos': frames[j]['pos'],
                'vel': frames[j]['vel'],
                'acc': frames[j]['acc'],
                'pos_next': frames[j+1]['pos'],
                'vel_next': frames[j+1]['vel'],
                'acc_next': frames[j+1]['acc']
            }
            dataset.append(sample)
    if shuffle:
        random.shuffle(dataset)

    return dataset


def generate_dataset(n_scenes=5, window_size = 2, shuffle=True):
    # Ensure the directory exists

    dataset = []
    
    print(f'Generating dataset with {n_scenes} scenes...')
    for _ in tqdm.tqdm(range(n_scenes)):
        scene = generate_scene_2gals()
        types= np.array(scene['types'])
        bh_index = np.where(types == "black hole")[0]
        frames = scene['frames']
        masses = scene['masses']
        for j in range(len(frames)-window_size):
            sample = {
                'masses': masses,
                'bh_index': bh_index,
                'pos': frames[j]['pos'],
                'vel': frames[j]['vel'],
                'acc': frames[j]['acc']
            }
            for k in range(1, window_size):
                sample['pos_next{}'.format(k)] = frames[j+k]['pos']
                sample['vel_next{}'.format(k)] = frames[j+k]['vel']
                sample['acc_next{}'.format(k)] = frames[j+k]['acc']
            dataset.append(sample)
    if shuffle:
        random.shuffle(dataset)

    return dataset

def generate_dataset_1gal(n_scenes=5, window_size=4, shuffle=True, particles=500):
    
    dataset = []
    
    print(f'Generating dataset with {n_scenes} scenes...')
    for _ in tqdm.tqdm(range(n_scenes)):
        scene = generate_scene_1gal(particles)
        types= np.array(scene['types'])
        frames = scene['frames'][25:]
        masses = scene['masses']
        for j in range(len(frames)-window_size):
            sample = {
                'masses': masses,
                'pos': frames[j]['pos'],
                'vel': frames[j]['vel'],
                'acc': frames[j]['acc']
            }
            for k in range(1, window_size):
                sample['pos_next{}'.format(k)] = frames[j+k]['pos']
                sample['vel_next{}'.format(k)] = frames[j+k]['vel']
                sample['acc_next{}'.format(k)] = frames[j+k]['acc']
            dataset.append(sample)
    if shuffle:
        random.shuffle(dataset)

    return dataset

def generate_dataset_past(n_scenes=5, window_size=4, shuffle=True):

        dataset = []
        print(f'Generating dataset with {n_scenes} scenes...')
        for _ in tqdm.tqdm(range(n_scenes)):
            scene = generate_scene_2gals()
            frames = scene['frames']
            masses = scene['masses']
            for j in range(window_size, len(frames)):
                sample = {
                    'masses': masses,
                    'pos': frames[j]['pos'],
                    'vel': frames[j]['vel'],
                    'acc': frames[j]['acc']
                }
                past_pos = []
                for k in range(1, window_size):
                    past_pos.append(np.array(frames[j-k]['pos']))
                    #sample['vel_past{}'.format(k)] = frames[j-k]['vel']
                    #sample['acc_past{}'.format(k)] = frames[j-k]['acc']

                sample['past_pos'] = np.concatenate(past_pos, axis=-1).tolist()
                dataset.append(sample)

        if shuffle:
            random.shuffle(dataset)

        return dataset

def generate_dataset_transformer(n_scenes=5, window_size=4, shuffle=True):

        dataset = []
        print(f'Generating dataset with {n_scenes} scenes...')
        for _ in tqdm.tqdm(range(n_scenes)):
            scene = generate_scene_2gals()
            frames = scene['frames']
            masses = np.array(scene['masses']).reshape(-1, 1)
            for j in range(window_size, len(frames)):
                sample = {}
                inputs = []
                for k in range(window_size):
                    time_step_pos = np.array(frames[j-window_size+k]['pos'])
                    time_stemp_vel = np.array(frames[j-window_size+k]['vel'])
                    time_step_masses = masses
                    time_step_input = np.concatenate([time_step_pos, time_stemp_vel, time_step_masses], axis=-1)
                    inputs.append(time_step_input)

                actual_pos = np.array(frames[j]['pos'])
                actual_vel = np.array(frames[j]['vel'])
                actual_masses = masses
                actual_input = np.concatenate([actual_pos, actual_vel, actual_masses], axis=-1)
                inputs.append(actual_input)
                accelerations = np.array(frames[j]['acc'])
                sample['inputs'] = np.array(inputs)
                sample['accelerations'] = accelerations
                dataset.append(sample)

        if shuffle:
            random.shuffle(dataset)

        return dataset



