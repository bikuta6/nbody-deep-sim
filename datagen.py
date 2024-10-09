from sim import NBodySimulation, generateDisk3Dv3
import numpy as np
import json
import glob
import os

def gen_params():
    return {
        'nbStars': int(np.random.randint(100, 500)),
        'radius': float(np.random.uniform(1.0, 2.0)),
        'Mass': float(np.random.uniform(1.0, 3.0)),
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
    print(params1)
    print(params2)

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

    return json.dumps(final_json, indent=4)


def generate_dataset(n_scenes=5, window_size=3, shuffle=True, dir='./train/', save=True):
    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)
    
    # the objective is to generate samples from n scenes, of 3 frames each, saving positions, velocities and accelerations, the idea is to predict acceleration from the first frame for the next 2 frames to integrate position and velocity
    other_files = glob.glob(dir + '*.json')
    # get the last id from the files, given the structure of the file names is train_0.json, train_1.json, etc
    last_id = -1
    for file in other_files:
        last_id = max(last_id, int(file.split('_')[-1].split('.')[0]))

    new_id = last_id + 1
    name = dir + f'{dir[1:-1]}_{new_id}.json'
    
    dataset = []
    for i in range(n_scenes):
        scene = generate_scene_2gals()
        scene = json.loads(scene)
        frames = scene['frames']
        masses = scene['masses']
        for j in range(len(frames)-window_size):
            sample = {
                'masses': masses,
                'pos': frames[j]['pos'],
                'vel': frames[j]['vel'],
            }
            for k in range(1, window_size):
                sample['pos_next{}'.format(k)] = frames[j+k]['pos']
                sample['vel_next{}'.format(k)] = frames[j+k]['vel']
            dataset.append(sample)
    if shuffle:
        np.random.shuffle(dataset)
    if save:
        with open(name, 'w') as f:
            json.dump(dataset, f)

    return dataset
