#!/usr/bin/env python3
import argparse
import csv
import itertools

import numpy as np

from galaxify import galaxies, simulation


def main():
    parser = argparse.ArgumentParser(
        description="Generación de dataset de simulaciones de galaxias"
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        nargs="+",
        required=True,
        help="Número de cuerpos en la galaxia (valen varios valores)",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="leapfrog",
        choices=["leapfrog", "euler"],
        required=True,
        help="Integrador a utilizar",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Fichero de salida (CSV)"
    )
    parser.add_argument(
        "--sim-type",
        type=str,
        nargs="+",
        choices=["disk", "spiral"],
        default=["disk"],
        help="Tipo de simulación",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Número de pasos de la simulación"
    )
    parser.add_argument("--dt", type=float, default=0.0001, help="Paso temporal dt")
    parser.add_argument(
        "--softening", type=float, default=0.05, help="Parámetro de suavizado"
    )
    parser.add_argument(
        "--g", type=float, default=4.5e-6, help="Constante gravitacional"
    )
    parser.add_argument(
        "--total-mass", type=float, default=1.0, help="Masa total de la galaxia"
    )
    parser.add_argument(
        "--radial-scale", type=float, default=3.0, help="Escala radial de la galaxia"
    )
    parser.add_argument(
        "--height-scale", type=float, default=0.3, help="Escala vertical de la galaxia"
    )
    parser.add_argument(
        "--black-hole-mass",
        type=float,
        default=0.01,
        help="Fracción de la masa total para el agujero negro (spiral)",
    )
    parser.add_argument(
        "--n-arms", type=int, default=2, help="Número de brazos espirales (spiral)"
    )
    parser.add_argument(
        "--pitch-angle",
        type=float,
        default=-np.pi / 6,
        help="Ángulo de pitch (en radianes, spiral).",
    )
    parser.add_argument(
        "--arm-strength",
        type=float,
        default=0.3,
        help="Perturbación angular de los brazos (spiral).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla para el generador aleatorio."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Dispositivo para simular",
    )
    args = parser.parse_args()

    params = {}
    for key, value in vars(args).items():
        if key in ["output", "device"]:
            continue
        if isinstance(value, list):
            params[key] = value
        else:
            params[key] = [value]

    keys = list(params.keys())
    combinations = list(itertools.product(*(params[k] for k in keys)))
    print(f"Generando {len(combinations)} escenarios...")

    print(f"Creando dataset {args.output}")
    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "scene",
            "scene_type",
            "step",
            "step_time",
            "mass",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "ax",
            "ay",
            "az",
            "u",
            "k",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        scene_id = 0
        for combo in combinations:
            combo_dict = dict(zip(keys, combo))
            n_bodies = combo_dict["n_bodies"]
            sim_type = combo_dict["sim_type"]
            steps = combo_dict["steps"]
            dt = combo_dict["dt"]
            softening = combo_dict["softening"]
            g_const = combo_dict["g"]
            black_hole_mass = combo_dict["black_hole_mass"]
            total_mass = combo_dict["total_mass"]
            radial_scale = combo_dict["radial_scale"]
            height_scale = combo_dict["height_scale"]
            seed = combo_dict["seed"]

            print("-" * 80)
            print(f"Escenario {scene_id + 1}/{len(combinations)}")
            print(f"  Número de cuerpos: {n_bodies}")
            print(f"  Tipo de simulación: {sim_type}")
            print(f"  Pasos de simulación: {steps}")
            print(f"  Paso temporal: {dt}")
            print(f"  Parámetro de suavizado: {softening}")
            print(f"  Constante gravitacional: {g_const}")
            print(f"  Masa total: {total_mass}")
            print(f"  Agujero negro: {black_hole_mass} de la masa total")
            print(f"  Escala radial: {radial_scale}")
            print(f"  Escala vertical: {height_scale}")
            print(f"  Semilla aleatoria: {seed}")

            print("Generando galaxia...", end="")
            if sim_type == "disk":
                pos, vel, masses = galaxies.generate_disk(
                    n_bodies=n_bodies,
                    total_mass=total_mass,
                    black_hole_mass=black_hole_mass,
                    radial_scale=radial_scale,
                    height_scale=height_scale,
                    g_const=g_const,
                    seed=seed,
                )
            elif sim_type == "spiral":
                black_hole_mass = combo_dict.get("black_hole_mass", 0.01)
                n_arms = combo_dict["n_arms"]
                pitch_angle = combo_dict["pitch_angle"]
                arm_strength = combo_dict["arm_strength"]
                pos, vel, masses = galaxies.generate_spiral(
                    n_bodies=n_bodies,
                    total_mass=total_mass,
                    radial_scale=radial_scale,
                    height_scale=height_scale,
                    g_const=g_const,
                    black_hole_mass=black_hole_mass,
                    n_arms=n_arms,
                    pitch_angle=pitch_angle,
                    arm_strength=arm_strength,
                    seed=seed,
                )
            else:
                raise ValueError(f"Tipo de simulación desconocido: {sim_type}")
            print("✅")

            print("Simulando...", end="")

            if args.integrator == "euler":
                simulator = simulation.EulerSimulator(
                    positions=pos,
                    velocities=vel,
                    masses=masses,
                    g_const=g_const,
                    softening=softening,
                    dt=dt,
                    calc_energy=True,
                    device=args.device,
                )
            else:
                simulator = simulation.LeapFrogSimulator(
                    positions=pos,
                    velocities=vel,
                    masses=masses,
                    g_const=g_const,
                    softening=softening,
                    dt=dt,
                    calc_energy=True,
                    device=args.device,
                )
            states = simulator.run(steps)
            print("✅")

            print("Escribiendo datos...", end="")
            for state in states:
                pos_arr = state.positions.cpu().numpy()
                vel_arr = state.velocities.cpu().numpy()
                acc_arr = state.accelerations.cpu().numpy()
                for i in range(pos_arr.shape[0]):
                    row = {
                        "scene": scene_id,
                        "scene_type": sim_type,
                        "step": state.step,
                        "step_time": state.step_time,
                        "mass": masses[i],
                        "x": pos_arr[i, 0],
                        "y": pos_arr[i, 1],
                        "z": pos_arr[i, 2],
                        "vx": vel_arr[i, 0],
                        "vy": vel_arr[i, 1],
                        "vz": vel_arr[i, 2],
                        "ax": acc_arr[i, 0],
                        "ay": acc_arr[i, 1],
                        "az": acc_arr[i, 2],
                        "u": state.u_energy,
                        "k": state.k_energy,
                    }
                    writer.writerow(row)
            scene_id += 1
            print("✅")


if __name__ == "__main__":
    main()
