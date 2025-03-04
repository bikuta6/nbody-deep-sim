import numpy as np
from matplotlib import animation, pyplot as plt

from galaxify import galaxies
from galaxify.simulation import LeapFrogSimulator

if __name__ == "__main__":
    seed = 42

    n_bodies = 5000

    radial_scale = 3.0  # kpc
    total_mass = 5e10  # Masas solares
    black_hole_mass = 0.01  # 1% de la galaxia
    height_scale = 0.3  # kpc
    g_const = 4.5e-6  # kpc^3/(M_sol·Myr^2)
    dt = 0.0001  # Myr
    simulation_steps = 6000
    n_arms = 2
    arm_strength = 0.3
    pitch_angle = -np.pi / 6
    softening = 0.05  # kpc
    calc_energy = False

    device = "cuda"
    '''

    positions, velocities, masses = galaxies.generate_disk(
        n_bodies=n_bodies,
        total_mass=total_mass,
        black_hole_mass=black_hole_mass,
        radial_scale=radial_scale,
        height_scale=height_scale,
        g_const=g_const,
        seed=seed,
    )
    '''
    positions, velocities, masses = galaxies.generate_spiral(
        n_bodies=n_bodies,
        total_mass=total_mass,
        black_hole_mass=black_hole_mass,
        radial_scale=radial_scale,
        height_scale=height_scale,
        g_const=g_const,
        n_arms=n_arms,
        arm_strength=arm_strength,
        seed=seed
    )

    # Simulaciones
    for simulator in [LeapFrogSimulator, ]:  # EulerSimulator, RK4Simulator]:  # TODO Meter más simuladores
        print("-" * 80)
        simulator_name = simulator.__name__.replace("Simulator", "")
        print(f"Simulación con {simulator_name}...", end="")

        sim = simulator(
            positions=positions,
            velocities=velocities,
            masses=masses,
            g_const=g_const,
            softening=softening,
            dt=dt,
            calc_energy=calc_energy,
            device=device,
        )
        memory = sim.run(simulation_steps)
        print("✅")

        last_state = memory[-1]
        total_time = sum(state.step_time for state in memory)
        print(f"Tiempo de simulación: {total_time:.2f} s")
        print(f"Tiempo de ejecución del último paso (s): {last_state.step_time:2f}")
        if last_state.u_energy is not None:
            print(f"Energía potencial (U): {last_state.u_energy:.2f}")
        if last_state.k_energy is not None:
            print(f"Energía cinética (K): {last_state.k_energy:.2f}")

        print("Generando vídeo... ", end="")
        fig, ax = plt.subplots(figsize=(16, 16))
        scatter = ax.scatter(memory[0].positions[:, 0], memory[0].positions[:, 1], s=1, c="black")
        ax.set_xlim(-max(memory[0].positions[:, 0]), max(memory[0].positions[:, 0]))
        ax.set_ylim(-max(memory[0].positions[:, 1]), max(memory[0].positions[:, 1]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Step 0")


        def update(frame):
            pos = memory[frame].positions
            scatter.set_offsets(pos[:, :2])
            ax.set_title(f"Step {frame}")
            return scatter,


        ani = animation.FuncAnimation(fig, update, frames=len(memory), interval=50, blit=True)
        ani.save(
            f"{device}-{simulator_name}-{n_bodies}bodies{last_state.step + 1}steps-{total_time:.2f}s.mp4",
            writer="ffmpeg",
            fps=120
        )
        print("✅")
