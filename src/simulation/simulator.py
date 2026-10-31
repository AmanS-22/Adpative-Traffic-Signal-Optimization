"""
Simulator - Main simulation engine

Key features:
- Accepts a pre-generated vehicle schedule for FAIR controller comparison
- Uses config.duration for correct throughput calculation
- Runs multiple seeds for statistical validity
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .vehicle import Vehicle, VehicleType
from .intersection import Intersection, SignalPhase
from .traffic_generator import TrafficGenerator, TrafficScenario


class ControllerType(Enum):
    FIXED = "fixed"
    RULE_BASED = "rule_based"
    DEEP_LEARNING = "deep_learning"


@dataclass
class SimulationConfig:
    duration: float = 300.0
    dt: float = 1.0
    scenario: TrafficScenario = TrafficScenario.NORMAL
    controller: ControllerType = ControllerType.RULE_BASED
    seed: Optional[int] = None
    fixed_green_time: float = 30.0
    rule_min_green: float = 10.0
    rule_max_green: float = 40.0
    rule_hysteresis: float = 0.2
    record_history: bool = True
    verbose: bool = False


@dataclass
class SimulationResult:
    config: SimulationConfig
    vehicles_served: int
    average_waiting_time: float
    max_waiting_time: float
    average_queue_length: float
    max_queue_length: float
    average_travel_time: float
    throughput: float
    phase_changes: int
    signal_utilization: float
    congestion_percentage: float
    vehicle_type_counts: Dict[str, int]
    queue_history: List[Dict]
    phase_history: List[Dict]
    timestamp: float = field(default_factory=time.time)


class Simulator:
    """
    Main simulation engine.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.intersection = Intersection()
        self.generator = TrafficGenerator(seed=self.config.seed)
        self.current_time = 0.0
        self.history: List[Dict] = []

        self._reset_statistics()
        self._setup_scenario()

    # ------------------------------------------------------------------
    def _reset_statistics(self):
        self.vehicles_served = 0
        self.total_waiting_time = 0.0
        self.total_travel_time = 0.0
        self.queue_lengths: List[Dict] = []
        self.phase_changes = 0
        self.green_time = 0.0
        self.vehicle_type_counts = {vt.value: 0 for vt in VehicleType}
        self.max_queue_observed = 0
        self._phase_elapsed = 0.0

    def _setup_scenario(self):
        self.generator.set_scenario(self.config.scenario)
        self.generator.apply_scenario_to_intersection(self.intersection)
        self.intersection.phase_start_time = 0.0

        if self.config.controller == ControllerType.RULE_BASED:
            self.intersection.min_green = self.config.rule_min_green
            self.intersection.max_green = self.config.rule_max_green

    # ------------------------------------------------------------------
    def _apply_controller_logic(self):
        """Apply the controller decision for the current step."""
        if self.config.controller == ControllerType.FIXED:
            self._phase_elapsed += self.config.dt
            if self._phase_elapsed >= self.config.fixed_green_time:
                new_phase = (SignalPhase.EW_GO
                             if self.intersection.phase == SignalPhase.NS_GO
                             else SignalPhase.NS_GO)
                self.intersection.switch_phase(new_phase, self.current_time)
                self._phase_elapsed = 0.0
                self.phase_changes += 1

        elif self.config.controller == ControllerType.RULE_BASED:
            if self.intersection.should_switch(self.current_time):
                current_demand = self.intersection.get_phase_demand(self.intersection.phase)
                other_phase = (SignalPhase.EW_GO
                               if self.intersection.phase == SignalPhase.NS_GO
                               else SignalPhase.NS_GO)
                other_demand = self.intersection.get_phase_demand(other_phase)

                should = (
                    other_demand > current_demand * (1 + self.config.rule_hysteresis)
                    or self.intersection.get_elapsed_green(self.current_time)
                       >= self.config.rule_max_green
                )
                if should:
                    self.intersection.switch_phase(other_phase, self.current_time)
                    self.phase_changes += 1

        elif self.config.controller == ControllerType.DEEP_LEARNING:
            # Instantiate the DL controller lazily (once)
            if not hasattr(self, '_dl_controller') or self._dl_controller is None:
                from src.controllers.dl_controller import DeepLearningController
                self._dl_controller = DeepLearningController(
                    intersection=self.intersection,
                    model_path="model/best_model.keras",
                    scaler_path="model/scaler.pkl",
                    min_green=self.config.rule_min_green,
                    max_green=self.config.rule_max_green,
                    confidence_threshold=0.40,
                    debug=True,
                )

            if self._dl_controller.decide(self.current_time):
                other_phase = (SignalPhase.EW_GO
                               if self.intersection.phase == SignalPhase.NS_GO
                               else SignalPhase.NS_GO)
                self.intersection.switch_phase(other_phase, self.current_time)
                self.phase_changes += 1
                self._dl_controller.reset()

    # ------------------------------------------------------------------
    def _collect_metrics(self):
        q = self.intersection.get_queue_lengths()
        total = sum(q.values())
        self.queue_lengths.append({
            'time': self.current_time,
            'total': total,
            'N': q['N'], 'S': q['S'], 'E': q['E'], 'W': q['W'],
        })
        if total > self.max_queue_observed:
            self.max_queue_observed = total

    def _update_statistics(self):
        self.vehicles_served = self.intersection.vehicles_served_total
        self.total_waiting_time = self.intersection.total_waiting_time
        for road in self.intersection.roads.values():
            for v in road.get_vehicles():
                self.vehicle_type_counts[v.vehicle_type.value] += 1

    # ------------------------------------------------------------------
    def run(self, schedule: Optional[Dict[int, Dict[str, List[Vehicle]]]] = None) -> SimulationResult:
        """
        Run the simulation.

        Args:
            schedule: Optional pre-generated schedule. If None, one is
                      generated here from self.config.seed. Passing the
                      SAME schedule to multiple controllers guarantees
                      identical traffic input.
        """
        self._reset_statistics()
        self.intersection.reset()
        self.generator.reset()
        self._setup_scenario()
        self.history = []
        self.current_time = 0.0
        self._phase_elapsed = 0.0
        self._dl_controller = None

        # Deterministic RNG if any randomness remains
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Pre-generate schedule if not provided
        if schedule is None:
            schedule = self.generator.generate_schedule(
                duration=self.config.duration,
                dt=self.config.dt,
            )

        print(f"🚦 Running {self.config.controller.value} with {self.config.scenario.value}...")

        step = 0
        while self.current_time < self.config.duration:
            # Add scheduled vehicles
            if step in schedule:
                for road, vehicles in schedule[step].items():
                    for v in vehicles:
                        self.intersection.add_vehicle(road, v)

            # Controller decision
            self._apply_controller_logic()

            # Advance simulation
            self.intersection.step(self.current_time, self.config.dt)

            # Metrics
            self._collect_metrics()
            self._update_statistics()

            self.current_time += self.config.dt
            step += 1

        return self._build_results()

    # ------------------------------------------------------------------
    def _build_results(self) -> SimulationResult:
        avg_queue = (float(np.mean([q['total'] for q in self.queue_lengths]))
                     if self.queue_lengths else 0.0)
        avg_wait = self.total_waiting_time / max(1, self.vehicles_served)

        # Correct throughput: served / duration * 3600
        if self.config.duration > 0:
            throughput = (self.vehicles_served / self.config.duration) * 3600
        else:
            throughput = 0.0

        # Signal utilization
        if self.config.controller == ControllerType.FIXED:
            total_green = self.phase_changes * self.config.fixed_green_time
            signal_utilization = (total_green / self.config.duration) * 100
        else:
            signal_utilization = 50.0

        # Congestion
        congestion_steps = sum(1 for q in self.queue_lengths if q['total'] > 10)
        congestion_percentage = (congestion_steps / max(1, len(self.queue_lengths))) * 100

        return SimulationResult(
            config=self.config,
            vehicles_served=self.vehicles_served,
            average_waiting_time=avg_wait,
            max_waiting_time=self.max_queue_observed * 0.5,
            average_queue_length=avg_queue,
            max_queue_length=self.max_queue_observed,
            average_travel_time=avg_wait * 0.5,
            throughput=throughput,
            phase_changes=self.phase_changes,
            signal_utilization=signal_utilization,
            congestion_percentage=congestion_percentage,
            vehicle_type_counts=self.vehicle_type_counts,
            queue_history=self.queue_lengths,
            phase_history=self.intersection.history,
        )

    # ------------------------------------------------------------------
    def run_multiple(self, num_runs: int = 5,
                     base_seed: Optional[int] = None) -> List[SimulationResult]:
        """
        Run multiple trials with DIFFERENT seeds, each with its own
        pre-generated schedule (identical to what other controllers
        would see if given the same seed).
        """
        results = []
        base = base_seed if base_seed is not None else (self.config.seed or 0)

        for i in range(num_runs):
            seed = base + i
            self.config.seed = seed
            self.generator = TrafficGenerator(seed=seed)
            self.generator.set_scenario(self.config.scenario)

            schedule = self.generator.generate_schedule(
                duration=self.config.duration,
                dt=self.config.dt,
            )
            result = self.run(schedule=schedule)
            results.append(result)
            print(f"  Run {i+1}/{num_runs} (seed={seed}): "
                  f"Served {result.vehicles_served}, "
                  f"Wait {result.average_waiting_time:.2f}s")

        return results

    # ------------------------------------------------------------------
    def print_results(self, result: SimulationResult):
        print("\n" + "=" * 60)
        print(f"📊 {result.config.controller.value} on {result.config.scenario.value}")
        print("=" * 60)
        print(f"Vehicles Served:      {result.vehicles_served}")
        print(f"Avg Waiting Time:     {result.average_waiting_time:.2f}s")
        print(f"Avg Queue Length:     {result.average_queue_length:.2f}")
        print(f"Max Queue Length:     {result.max_queue_length}")
        print(f"Phase Changes:        {result.phase_changes}")
        print(f"Throughput:           {result.throughput:.1f} veh/h")
        print(f"Congestion:           {result.congestion_percentage:.1f}%")
        print("=" * 60)