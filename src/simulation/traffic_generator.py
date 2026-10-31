"""
Traffic Generator - Creates realistic traffic patterns for simulation

Includes deterministic schedule pre-generation so that ALL controllers
(Fixed, Rule-Based, Deep Learning) receive EXACTLY the same traffic.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from .vehicle import Vehicle, VehicleType
from .intersection import Intersection


class TrafficScenario(Enum):
    """Traffic scenarios for simulation"""
    NORMAL = "normal"
    MORNING_RUSH = "morning_rush"
    EVENING_RUSH = "evening_rush"
    BALANCED = "balanced"
    HEAVY_NS = "heavy_ns"
    HEAVY_EW = "heavy_ew"
    SUDDEN_SPIKE = "sudden_spike"
    LOW_TRAFFIC = "low_traffic"

    @property
    def description(self) -> str:
        return {
            TrafficScenario.NORMAL: "Normal balanced traffic",
            TrafficScenario.MORNING_RUSH: "Morning rush hour (8-9 AM)",
            TrafficScenario.EVENING_RUSH: "Evening rush hour (5-6 PM)",
            TrafficScenario.BALANCED: "Balanced traffic all directions",
            TrafficScenario.HEAVY_NS: "Heavy North-South traffic",
            TrafficScenario.HEAVY_EW: "Heavy East-West traffic",
            TrafficScenario.SUDDEN_SPIKE: "Sudden traffic spike",
            TrafficScenario.LOW_TRAFFIC: "Low traffic volume",
        }[self]


@dataclass
class TrafficPattern:
    """Traffic pattern configuration"""
    name: str
    base_rates: Dict[str, Tuple[float, float]]
    vehicle_mix: Dict[VehicleType, float]
    duration: float
    peak_factor: float = 1.0
    variation: float = 0.2


class TrafficGenerator:
    """
    Generates realistic traffic for simulation.

    Key feature: `generate_schedule()` produces the COMPLETE vehicle
    arrival timeline deterministically from a seed. Reusing this schedule
    for all controllers guarantees fair comparison.
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        # Dedicated RNG instances so global random state doesn't interfere
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._np_rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        self.current_scenario = TrafficScenario.NORMAL
        self.current_time = 0.0
        self.is_active = True
        self.pattern = self._get_pattern(TrafficScenario.NORMAL)

    # ------------------------------------------------------------------
    def _get_pattern(self, scenario: TrafficScenario) -> TrafficPattern:
        patterns = {
            TrafficScenario.NORMAL: TrafficPattern(
                name="Normal",
                base_rates={
                    'N': (0.2, 0.6), 'S': (0.2, 0.6),
                    'E': (0.2, 0.6), 'W': (0.2, 0.6),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.75, VehicleType.BIKE: 0.10,
                    VehicleType.BUS: 0.10, VehicleType.TRUCK: 0.05,
                },
                duration=300.0, peak_factor=1.0,
            ),
            TrafficScenario.MORNING_RUSH: TrafficPattern(
                name="Morning Rush",
                base_rates={
                    'N': (0.8, 2.0), 'S': (0.3, 0.8),
                    'E': (0.6, 1.5), 'W': (0.4, 1.0),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.80, VehicleType.BIKE: 0.05,
                    VehicleType.BUS: 0.12, VehicleType.TRUCK: 0.03,
                },
                duration=600.0, peak_factor=2.0,
            ),
            TrafficScenario.EVENING_RUSH: TrafficPattern(
                name="Evening Rush",
                base_rates={
                    'N': (0.3, 0.8), 'S': (0.8, 2.0),
                    'E': (0.4, 1.0), 'W': (0.6, 1.5),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.80, VehicleType.BIKE: 0.05,
                    VehicleType.BUS: 0.12, VehicleType.TRUCK: 0.03,
                },
                duration=600.0, peak_factor=2.0,
            ),
            TrafficScenario.BALANCED: TrafficPattern(
                name="Balanced",
                base_rates={
                    'N': (0.4, 0.8), 'S': (0.4, 0.8),
                    'E': (0.4, 0.8), 'W': (0.4, 0.8),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.70, VehicleType.BIKE: 0.12,
                    VehicleType.BUS: 0.10, VehicleType.TRUCK: 0.08,
                },
                duration=300.0, peak_factor=1.5,
            ),
            TrafficScenario.HEAVY_NS: TrafficPattern(
                name="Heavy NS",
                base_rates={
                    'N': (1.0, 2.5), 'S': (1.0, 2.5),
                    'E': (0.1, 0.3), 'W': (0.1, 0.3),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.75, VehicleType.BIKE: 0.08,
                    VehicleType.BUS: 0.10, VehicleType.TRUCK: 0.07,
                },
                duration=300.0, peak_factor=1.8,
            ),
            TrafficScenario.HEAVY_EW: TrafficPattern(
                name="Heavy EW",
                base_rates={
                    'N': (0.1, 0.3), 'S': (0.1, 0.3),
                    'E': (1.0, 2.5), 'W': (1.0, 2.5),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.75, VehicleType.BIKE: 0.08,
                    VehicleType.BUS: 0.10, VehicleType.TRUCK: 0.07,
                },
                duration=300.0, peak_factor=1.8,
            ),
            TrafficScenario.SUDDEN_SPIKE: TrafficPattern(
                name="Sudden Spike",
                base_rates={
                    'N': (0.2, 0.5), 'S': (0.2, 0.5),
                    'E': (0.2, 0.5), 'W': (0.2, 0.5),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.70, VehicleType.BIKE: 0.10,
                    VehicleType.BUS: 0.10, VehicleType.TRUCK: 0.10,
                },
                duration=600.0, peak_factor=1.0,
            ),
            TrafficScenario.LOW_TRAFFIC: TrafficPattern(
                name="Low Traffic",
                base_rates={
                    'N': (0.05, 0.15), 'S': (0.05, 0.15),
                    'E': (0.05, 0.15), 'W': (0.05, 0.15),
                },
                vehicle_mix={
                    VehicleType.CAR: 0.65, VehicleType.BIKE: 0.15,
                    VehicleType.BUS: 0.05, VehicleType.TRUCK: 0.15,
                },
                duration=300.0, peak_factor=0.5,
            ),
        }
        return patterns[scenario]

    # ------------------------------------------------------------------
    def set_scenario(self, scenario: TrafficScenario):
        self.current_scenario = scenario
        self.pattern = self._get_pattern(scenario)
        self.current_time = 0.0

    # ------------------------------------------------------------------
    def _random_vehicle_type(self) -> VehicleType:
        types = list(self.pattern.vehicle_mix.keys())
        weights = list(self.pattern.vehicle_mix.values())
        return self._rng.choices(types, weights=weights)[0]

    # ------------------------------------------------------------------
    def generate_schedule(
        self,
        duration: float,
        dt: float = 1.0,
    ) -> Dict[int, Dict[str, List[Vehicle]]]:
        """
        Pre-generate the COMPLETE vehicle arrival schedule deterministically.

        Returns a dict keyed by simulation step:
            schedule[step_index][road_name] = [Vehicle, Vehicle, ...]

        Using this schedule for all controllers guarantees identical traffic.
        """
        # Re-seed internal RNGs so calls are reproducible
        if self._seed is not None:
            self._rng = random.Random(self._seed)
            self._np_rng = np.random.RandomState(self._seed)

        schedule: Dict[int, Dict[str, List[Vehicle]]] = {}
        num_steps = int(duration / dt)

        for step in range(num_steps):
            t = step * dt
            schedule[step] = self._generate_at_time(t, dt)

        return schedule

    # ------------------------------------------------------------------
    def _generate_at_time(self, current_time: float, dt: float) -> Dict[str, List[Vehicle]]:
        """Generate arrivals for one timestep (deterministic via self._rng)."""
        if self.current_scenario == TrafficScenario.SUDDEN_SPIKE:
            return self._generate_sudden_spike(current_time, dt)

        new_vehicles = {road: [] for road in ['N', 'S', 'E', 'W']}

        for road, (min_rate, max_rate) in self.pattern.base_rates.items():
            base_rate = self._rng.uniform(min_rate, max_rate)
            rate = base_rate * self.pattern.peak_factor
            variation = self._rng.uniform(
                1 - self.pattern.variation,
                1 + self.pattern.variation,
            )
            rate = max(0.0, rate * variation)

            num = self._np_rng.poisson(rate * dt)
            for _ in range(num):
                vtype = self._random_vehicle_type()
                new_vehicles[road].append(Vehicle(
                    vehicle_type=vtype,
                    arrival_time=current_time,
                    road=road,
                ))

        return new_vehicles

    def _generate_sudden_spike(self, current_time: float, dt: float) -> Dict[str, List[Vehicle]]:
        new_vehicles = {road: [] for road in ['N', 'S', 'E', 'W']}
        spike_start, spike_end, spike_duration = 100.0, 150.0, 50.0

        for road, (min_rate, max_rate) in self.pattern.base_rates.items():
            base_rate = self._rng.uniform(min_rate, max_rate)
            if spike_start <= current_time <= spike_end:
                progress = (current_time - spike_start) / spike_duration
                spike_factor = np.exp(-((progress - 0.5) * 6) ** 2) * 4.0
                rate = base_rate * (1 + spike_factor)
            else:
                rate = base_rate * 0.5

            rate *= self._rng.uniform(0.8, 1.2)
            rate = max(0.0, rate)

            num = self._np_rng.poisson(rate * dt)
            for _ in range(num):
                vtype = self._random_vehicle_type()
                new_vehicles[road].append(Vehicle(
                    vehicle_type=vtype,
                    arrival_time=current_time,
                    road=road,
                ))

        return new_vehicles

    # ------------------------------------------------------------------
    def generate_vehicles(self, current_time: float, dt: float = 1.0) -> Dict[str, List[Vehicle]]:
        """Legacy per-step generation (kept for backwards compatibility)."""
        self.current_time = current_time
        return self._generate_at_time(current_time, dt)

    # ------------------------------------------------------------------
    def apply_scenario_to_intersection(self, intersection: Intersection):
        if self.current_scenario in [TrafficScenario.MORNING_RUSH, TrafficScenario.EVENING_RUSH]:
            intersection.min_green = 15.0
            intersection.max_green = 50.0
        elif self.current_scenario in [TrafficScenario.HEAVY_NS, TrafficScenario.HEAVY_EW]:
            intersection.min_green = 12.0
            intersection.max_green = 45.0
        elif self.current_scenario == TrafficScenario.LOW_TRAFFIC:
            intersection.min_green = 5.0
            intersection.max_green = 20.0
        else:
            intersection.min_green = 10.0
            intersection.max_green = 40.0

    def get_statistics(self) -> Dict:
        return {
            'scenario': self.current_scenario.value,
            'scenario_description': self.current_scenario.description,
            'time': self.current_time,
            'peak_factor': self.pattern.peak_factor,
            'active': self.is_active,
        }

    def get_scenario_info(self) -> Dict[str, str]:
        return {s.value: s.description for s in TrafficScenario}

    def reset(self):
        self.current_time = 0.0
        self.is_active = True

    def __repr__(self) -> str:
        return f"TrafficGenerator(scenario={self.current_scenario.value}, time={self.current_time:.1f}s)"