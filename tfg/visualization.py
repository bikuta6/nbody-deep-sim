import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_scene(positions):
    
    positions = positions.cpu()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    def update(i):
        ax.clear()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.scatter(positions[i, :, 0], positions[i, :, 1], positions[i, :, 2])

    ani = FuncAnimation(fig, update, frames=range(positions.shape[0]), interval=10)
    return ani