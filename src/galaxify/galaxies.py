import enum

import numpy as np


class BodyType(enum.Enum):
    BLACK_HOLE = "black hole"
    STAR = "star"


def spherical_hernquist_distribution(
        *,
        r: float | np.ndarray,
        r0: float = 1,
        total_mass: float = 1,
        avoid_distance_zero: bool = True,
) -> float | np.ndarray:
    """Calcula la contribución de masa de cada partícula de acuerdo al perfil esférico de Hernquist.

   La densidad se define como:
        ρ(r) = (total_mass / (2π)) * (r0 / (r * (r0 + r)**3))

    Siendo 'r' la distancia radial al centro de la galaxia, 'r0' el radio de escala y 'total_mass' la masa total.

    Ejemplo:

    >>> import numpy as np
    >>> r_values = np.array([0.0, 1.0, 2.0])
    >>> spherical_hernquist_distribution(r_values, r0=1, total_mass=1)
    array([ ... , ... , ...])

    :param r: Distancia(s) radial(es) al centro de la galaxia.
    :param r0: Radio total de escala, esto es, el límite entre la región central (donde la densidad es muy alta) y la
        región exterior (donde la densidad decrece más rápidamente). Por defecto es 1.
    :param total_mass: Masa total de la galaxia. Por defecto es 1.
    :param avoid_distance_zero: Si es True, se reemplazan los valores de r iguales a cero por un valor pequeño
        (np.finfo(np.float32).eps) para evitar la singularidad. Si es False, se lanzará un ValueError si se encuentra un
        valor cero. Por defecto es True.
    :return: Densidad de la partícula en la posición 'r'.
    :raises ValueError: Si 'r' es cero y avoid_distance_zero es False.
    """
    r = np.asarray(r)

    if avoid_distance_zero:
        r = np.where(r == 0, np.finfo(np.float32).eps, r)
    else:
        if np.any(r == 0):
            raise ValueError("r contiene cero(s) y avoid_distance_zero es False")

    density = (total_mass / (2 * np.pi)) * (r0 / (r * (r0 + r) ** 3))
    return density


def generate_disk(
        *,
        n_bodies: int,
        total_mass: float,
        radial_scale: float,
        height_scale: float,
        g_const: float,
        black_hole_mass: float,
        offset=(0, 0, 0),
        initial_vel=(0, 0, 0),
        clockwise=True,
        angle=(0, 0, 0),
        seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Galaxia de tipo disco con un agujero negro en el centro.

    En esta versión modificada se asigna al agujero negro central una fracción de la masa total de la galaxia
    determinada por el parámetro 'black_hole_mass', de forma similar a la función 'generate_spiral'.

    :param n_bodies: Número de cuerpos a generar (incluyendo el agujero negro central).
    :param total_mass: Masa total de la galaxia. Las masas individuales se establecen de manera que su suma es esta.
    :param radial_scale: Escala radial (R_d) del disco exponencial.
    :param height_scale: Escala vertical del disco.
    :param g_const: Constante gravitacional, usada para calcular las velocidades de los cuerpos.
    :param black_hole_mass: Fracción de la masa total asignada al agujero negro central.
    :param offset: Posición a añadir a las posiciones generadas. Por defecto es (0, 0, 0).
    :param initial_vel: Velocidad inicial a añadir a las velocidades generadas. Por defecto es (0, 0, 0).
    :param clockwise: Si se invierte el sentido de giro. Por defecto es True.
    :param angle: Ángulos de Euler (en radianes) para la rotación de la galaxia. Por defecto es (0, 0, 0) (sin rotación).
    :param seed: Semilla para el generador de números aleatorios. Por defecto es None.
    :return: Una tupla con tres elementos:
        - positions: array de shape (n_bodies, 3) con las posiciones (x, y, z) de cada cuerpo.
        - velocities: array de shape (n_bodies, 3) con las velocidades (vx, vy, vz) de cada cuerpo.
        - masses: array de shape (n_bodies,) con las masas de cada cuerpo.
    """
    np.random.seed(seed)

    # Generamos los tipos de cuerpos: el primero es un agujero negro, el resto son estrellas.
    types = np.array([BodyType.STAR] * n_bodies)
    types[0] = BodyType.BLACK_HOLE

    ##############
    # POSICIONES #
    ##############
    # Generamos las distancias radiales con una transformación que favorece las regiones internas.
    distances = -radial_scale * np.log(1 - np.random.uniform(low=np.finfo(np.float32).eps, high=1.0, size=n_bodies))
    # El agujero negro se sitúa en el centro.
    distances[types == BodyType.BLACK_HOLE] = 0

    # Generamos la altura del disco, reduciéndose al acercarse al borde.
    zs = np.random.uniform(-1.0, 1.0, size=n_bodies) * height_scale * (1 - np.sqrt(distances))
    zs[types == BodyType.BLACK_HOLE] = 0

    # Ángulos aleatorios para distribuir las estrellas a lo largo del disco.
    phi = np.random.rand(n_bodies) * 2 * np.pi

    # Convertimos de coordenadas polares a cartesianas.
    xs = np.cos(phi) * distances
    ys = np.sin(phi) * distances
    positions = np.array((xs, ys, zs)).T

    #########
    # MASAS #
    #########
    # Se asigna la masa al agujero negro central como una fracción de la masa total.
    mass_bh = total_mass * black_hole_mass
    masses = np.empty(n_bodies)
    masses[0] = mass_bh

    # Para las estrellas se utiliza la distribución esférica de Hernquist para calcular pesos relativos.
    star_indices = (types == BodyType.STAR)
    # Se utiliza la distribución de Hernquist sobre las distancias de las estrellas.
    star_weights = spherical_hernquist_distribution(r=distances[star_indices], r0=1, total_mass=total_mass)
    # Se normalizan los pesos para que la suma de las masas de las estrellas sea (total_mass - mass_bh).
    star_weights_sum = star_weights.sum()
    masses[star_indices] = star_weights * ((total_mass - mass_bh) / star_weights_sum)

    ##############
    # VELOCIDADES #
    ##############
    velocities = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        if types[i] != BodyType.BLACK_HOLE:
            # Calculamos la masa total encerrada en la órbita de la estrella i.
            mass_enc = masses[distances < distances[i]].sum()
            # Estimamos la velocidad orbital circular.
            v = np.sqrt(g_const * mass_enc / distances[i])
            # Determinamos los componentes tangenciales de la velocidad.
            velocities[i, 0] = v * np.cos(phi[i] + np.pi / 2)
            velocities[i, 1] = v * np.sin(phi[i] + np.pi / 2)
            velocities[i, 2] = 0.0

    # Si 'clockwise' es True, invertimos la dirección del giro.
    if clockwise:
        velocities[:, 0] = -velocities[:, 0]
        velocities[:, 1] = -velocities[:, 1]

    # Creamos las matrices de rotación a partir de los ángulos de Euler proporcionados.
    euler_angles = np.array(angle)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
        [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])]
    ])
    ry = np.array([
        [np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
        [0, 1, 0],
        [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])]
    ])
    rz = np.array([
        [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
        [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
        [0, 0, 1]
    ])

    # Aplicamos las rotaciones a todas las posiciones y velocidades.
    positions = positions @ rx.T @ ry.T @ rz.T
    velocities = velocities @ rx.T @ ry.T @ rz.T

    # Trasladamos la galaxia al offset indicado.
    positions += np.array(offset)
    # Sumamos las velocidades iniciales proporcionadas.
    velocities += np.array(initial_vel)

    return positions, velocities, masses


def generate_spiral(
        *,
        n_bodies: int,
        total_mass: float,
        radial_scale: float,
        height_scale: float,
        g_const: float,
        black_hole_mass: float,
        n_arms: int = 2,
        pitch_angle: float = -np.pi / 6,
        arm_strength: float = 0.3,
        seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Galaxia espiral con un agujero negro en el centro.

    :param n_bodies: Número de partículas a generar (incluyendo el agujero negro central).
    :param total_mass: Masa total de la galaxia.
    :param radial_scale: Escala radial del disco exponencial.
    :param height_scale: Escala vertical del disco.
    :param g_const: Constante gravitacional, usada para calcular las velocidades.
    :param black_hole_mass: Fracción de la masa total asignada al agujero negro central.
    :param n_arms: Número de brazos espirales (por defecto 2).
    :param pitch_angle: Ángulo de pitch de los brazos (en radianes, por defecto -π/6).
    :param arm_strength: Factor de perturbación angular para acentuar los brazos espirales (por defecto 0.3).
    :param seed: Semilla para el generador de números aleatorios. Por defecto es None.
    :return: Una tupla con tres elementos: un array de forma (n_bodies, 3) con las posiciones (x, y, z) de cada cuerpo,
        un array de forma (n_bodies, 3) con las velocidades (vx, vy, vz) de cada cuerpo y un array de forma (n_bodies,)
        con las masas de cada cuerpo.
    """
    np.random.seed(seed)

    # Generamos los tipos de cuerpos: el primero es un agujero negro y el resto estrellas
    types = np.array([BodyType.STAR] * n_bodies)
    types[0] = BodyType.BLACK_HOLE

    # Inicializamos arrays de posiciones y velocidades
    positions = np.zeros((n_bodies, 3))
    velocities = np.zeros((n_bodies, 3))

    #########
    # MASAS #
    #########
    # Se asigna la masa del agujero negro central y se distribuye la masa restante uniformemente entre las estrellas
    mass_bh = total_mass * black_hole_mass
    masses = np.empty(n_bodies)
    masses[0] = mass_bh
    if n_bodies > 1:
        masses[1:] = (total_mass - mass_bh) / (n_bodies - 1)

    # Generamos posiciones y velocidades para cada cuerpo
    for i in range(n_bodies):
        if types[i] == BodyType.BLACK_HOLE:
            # El agujero negro se sitúa en el centro sin velocidad
            positions[i] = np.array([0.0, 0.0, 0.0])
            velocities[i] = np.array([0.0, 0.0, 0.0])
        else:
            ##############
            # POSICIONES #
            ##############
            # Se muestrea el radio 'r' usando una distribución Gamma (forma=2, escala=radial_scale)
            r = np.random.gamma(shape=2, scale=radial_scale)
            # Se muestrea el ángulo azimutal 'phi' uniformemente en [0, 2π)
            phi = 2 * np.pi * np.random.rand()
            # Se añade una perturbación espiral para acentuar la formación de brazos
            phi_spiral = phi + arm_strength * np.sin(
                n_arms * (phi - np.log(r / radial_scale) / np.tan(pitch_angle))) if r > 0 else phi

            # Coordenadas cartesianas: el disco se sitúa en el plano XY y la coordenada 'z' se asigna de forma gaussiana
            x = r * np.cos(phi_spiral)
            y = r * np.sin(phi_spiral)
            z = np.random.normal(0, height_scale)
            positions[i] = np.array([x, y, z])

            ###############
            # VELOCIDADES #
            ###############
            # Se estima la velocidad circular usando la masa encerrada en 'r' para un disco exponencial
            m_enc = total_mass * (1 - np.exp(-r / radial_scale) * (1 + r / radial_scale))
            v_circ = 0.0 if r < 1e-8 else np.sqrt(g_const * m_enc / r)
            sigma_r = 0.1 * v_circ
            sigma_phi = 0.07 * v_circ
            sigma_z = 0.05 * v_circ

            v_R = np.random.normal(0, sigma_r)
            v_phi = v_circ + np.random.normal(0, sigma_phi)
            v_z = np.random.normal(0, sigma_z)

            # Transformación a coordenadas cartesianas
            v_x = v_R * np.cos(phi_spiral) - v_phi * np.sin(phi_spiral)
            v_y = v_R * np.sin(phi_spiral) + v_phi * np.cos(phi_spiral)
            velocities[i] = np.array([v_x, v_y, v_z])

    return positions, velocities, masses
