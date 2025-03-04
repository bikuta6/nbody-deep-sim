import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SimulationState:
    """Estado de la simulación en un instante dado."""
    step: int  # Número de paso de la simulación
    step_time: float  # Tiempo que tomó ejecutar este paso (en segundos)
    positions: torch.Tensor  # Posiciones de las partículas (n_bodies, 3)
    velocities: torch.Tensor  # Velocidades de las partículas (n_bodies, 3)
    accelerations: torch.Tensor  # Aceleraciones de las partículas (n_bodies, 3)
    u_energy: float = None  # Energía potencial total del sistema (si se calculó)
    k_energy: float = None  # Energía cinética total del sistema (si se calculó)


class BaseSimulator:
    def __init__(
            self, *,
            positions: np.ndarray | torch.Tensor,
            velocities: np.ndarray | torch.Tensor,
            masses: np.ndarray | torch.Tensor,
            g_const: float = 1.0,
            softening: float = 0.1,
            dt: float = 0.01,
            calc_energy: bool = True,
            device: str = None,
    ):
        """Inicializa el simulador.

        :param positions: Posiciones iniciales (array o tensor de shape (n_bodies, 3)).
        :param velocities: Velocidades iniciales (array o tensor de shape (n_bodies, 3)).
        :param masses: Masas de las partículas (array o tensor de shape (n_bodies,)).
        :param g_const: Constante gravitacional.
        :param softening: Parámetro de suavizado para evitar singularidades en el cálculo de fuerzas.
        :param dt: Intervalo de tiempo para la integración.
        :param calc_energy: Si True, se calcularán las energías potencial y cinética en cada paso.
        :param device: 'cuda' o 'cpu'. Si es None se selecciona 'cuda' si está disponible, de lo contrario 'cpu'.
        :raises ValueError: Si device no es 'cuda', 'cpu' o None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device in ["cuda", "cpu"]:
            self.device = torch.device(device)
        else:
            raise ValueError("device debe ser 'cuda', 'cpu' o None")

        self.dt = dt
        self.g_const = g_const
        self.softening = softening
        self.calc_energy = calc_energy

        self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        self.velocities = torch.tensor(velocities, dtype=torch.float32, device=self.device)
        self.accelerations = None
        self.masses = torch.tensor(masses, dtype=torch.float32, device=self.device)

        self.n = self.positions.shape[0]

        self.accelerations = self.compute_accelerations()

    def compute_accelerations(self):
        """Calcula las aceleraciones gravitacionales para cada cuerpo usando la ley de gravitación de Newton.

        La aceleración de la partícula i es:
            a_i = G * sum_{j≠i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + softening^2)^(3/2)

        :return: Tensor de aceleraciones (n_bodies, 3).
        """
        # Calculamos las diferencias: diff[i,j] = r_j - r_i
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        # Distancias al cuadrado con suavizado
        dist_sq = (diff ** 2).sum(dim=2) + self.softening ** 2
        inv_dist_cube = dist_sq.pow(-1.5)
        # Queremos solo i != j
        inv_dist_cube.fill_diagonal_(0)
        acc = self.g_const * (diff * inv_dist_cube.unsqueeze(2) * self.masses.unsqueeze(0).unsqueeze(2)).sum(dim=1)
        return acc

    def compute_energies(self):
        """Calcula la energía potencial y cinética total del sistema.

        La energía potencial se calcula mediante el doble sumatorio sobre pares de partículas y la energía cinética a
        partir de la suma de 1/2 * m * v².

        :return: Tupla con la energía potencial y cinética del sistema.
        """
        # Energía cinética
        kinetic = 0.5 * self.masses * (self.velocities ** 2).sum(dim=1)
        k_energy = kinetic.sum().item()

        # Energía potencial
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        dist = (diff ** 2).sum(dim=2).sqrt() + self.softening
        # Evitamos la división por cero (tb. auto-interacción)
        mask = torch.eye(self.n, device=self.device, dtype=torch.bool)
        dist.masked_fill_(mask, float("inf"))
        potential_matrix = -self.g_const * (self.masses.unsqueeze(0) * self.masses.unsqueeze(1)) / dist
        # Sumamos solo por encima de la diagonal para no contar dos veces
        u_energy = potential_matrix.triu(1).sum().item()

        return u_energy, k_energy

    def run(self, steps: int) -> list[SimulationState]:
        """Ejecuta la simulación durante un número concreto de pasos.

        La ejecución almacenará en la memoria el estado de cada step de la simulación como objeto de

        :param steps: El número de pasos de simulación a ejecutar.
        :return:Lista de estados, donde cada estado es un objeto `SimulationStep`
        """
        states = []
        for step in range(steps):
            start_time = time.time()
            self.step()
            step_time = time.time() - start_time

            u_energy, k_energy = self.compute_energies() if self.calc_energy else (None, None)

            states.append(SimulationState(
                positions=self.positions.clone().cpu(),
                velocities=self.velocities.clone().cpu(),
                accelerations=self.accelerations.clone().cpu(),
                step=step,
                step_time=step_time,
                u_energy=u_energy,
                k_energy=k_energy,
            ))
        return states

    def step(self):
        """Avanza la simulación un paso de integración."""
        raise NotImplementedError("El método step debe ser implementado en la subclase")


class LeapFrogSimulator(BaseSimulator):
    def step(self):
        """Avanza la simulación un paso utilizando el esquema Leapfrog..

        El algoritmo Leap-Frog o Kick-Drift-Kick:
            1. v(t + dt/2) = v(t) + (dt/2) * a(t)
            2. x(t + dt) = x(t) + dt * v(t + dt/2)
            3. Calcular a(t + dt) a partir de x(t + dt)
            4. v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
        """
        # Calculamos las velocidades en medio paso...
        self.velocities += 0.5 * self.dt * self.accelerations
        # ...luego las posiciones con esas velocidades...
        self.positions += self.dt * self.velocities
        # ...luego las nuevas aceleraciones en las posiciones actualizadas...
        self.accelerations = self.compute_accelerations()
        # ...y por último actualizamos de nuevo las posiciones
        self.velocities += 0.5 * self.dt * self.accelerations


class EulerSimulator(BaseSimulator):
    def step(self):
        """Avanza la simulación un paso utilizando el esquema de Euler.

        El algoritmo de Euler:
            1. Calcular a(t)
            2. v(t + dt) = v(t) + dt * a(t)
            3. x(t + dt) = x(t) + dt * v(t)
        """
        # Calculamos las aceleraciones en las posiciones actuales...
        self.accelerations = self.compute_accelerations()
        # ...luego las velocidades...
        self.velocities += self.dt * self.accelerations
        # ...y por último las posiciones
        self.positions += self.dt * self.velocities


class RK4Simulator(BaseSimulator):
    def _compute_accelerations_at(self, positions: torch.Tensor) -> torch.Tensor:
        """Calcula las aceleraciones gravitacionales para un conjunto arbitrario de posiciones.

        Similar a `compute_accelerations` pero permite evaluar la aceleración en posiciones intermedias.

        :param positions: Tensor de posiciones (n_bodies, 3).
        :return: Tensor de aceleraciones (n_bodies, 3).
        """
        # Diferencias de posiciones entre cada par de cuerpos
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (n, n, 3)
        # Distancias al cuadrado con suavizado
        dist_sq = (diff ** 2).sum(dim=2) + self.softening ** 2  # (n, n)
        # Inversa del cubo de las distancias
        inv_dist_cube = dist_sq.pow(-1.5)
        # Evitamos la auto-interacción
        inv_dist_cube.fill_diagonal_(0)
        # Calculamos la aceleración total sobre cada partícula
        acc = self.g_const * (diff * inv_dist_cube.unsqueeze(2) * self.masses.unsqueeze(0).unsqueeze(2)).sum(dim=1)
        return acc

    def step(self):
        """Avanza la simulación un paso utilizando el esquema de integración Runge-Kutta de 4º orden (RK4).

        El algoritmo RK4 se aplica sobre el sistema:
            d(pos)/dt = velocidad
            d(velocidad)/dt = aceleración(posiciones)

        De forma concreta:
            1. k1 = f(y)         donde f(y) = [velocidades, aceleraciones( posiciones )]
            2. k2 = f(y + dt/2*k1)
            3. k3 = f(y + dt/2*k2)
            4. k4 = f(y + dt*k3)
            5. y(t+dt) = y(t) + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        """
        dt = self.dt

        # k1: evaluamos en el estado actual
        k1_pos = self.velocities
        k1_vel = self.compute_accelerations()

        # k2: evaluamos en el estado intermedio (dt/2)
        pos_k2 = self.positions + 0.5 * dt * k1_pos
        vel_k2 = self.velocities + 0.5 * dt * k1_vel
        k2_pos = vel_k2
        k2_vel = self._compute_accelerations_at(pos_k2)

        # k3: evaluamos nuevamente en un estado intermedio (dt/2)
        pos_k3 = self.positions + 0.5 * dt * k2_pos
        vel_k3 = self.velocities + 0.5 * dt * k2_vel
        k3_pos = vel_k3
        k3_vel = self._compute_accelerations_at(pos_k3)

        # k4: evaluamos en el estado final (dt)
        pos_k4 = self.positions + dt * k3_pos
        vel_k4 = self.velocities + dt * k3_vel
        k4_pos = vel_k4
        k4_vel = self._compute_accelerations_at(pos_k4)

        # Actualizamos las posiciones y velocidades combinando los incrementos
        self.positions = self.positions + (dt / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        self.velocities = self.velocities + (dt / 6) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        # Recalculamos las aceleraciones a partir de las nuevas posiciones
        self.accelerations = self.compute_accelerations()
