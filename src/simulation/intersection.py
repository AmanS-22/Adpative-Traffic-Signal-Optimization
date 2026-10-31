"""
Intersection Module - 4-way intersection with traffic signals

IMPORTANT: All time values are SIMULATION time (seconds from start),
NOT wall-clock time. This ensures reproducibility and correctness.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from .road import Road
from .vehicle import Vehicle, VehicleType


class SignalPhase(Enum):
    """Traffic signal phases"""
    NS_GO = "NS_GO"
    EW_GO = "EW_GO"
    ALL_RED = "ALL_RED"

    @property
    def green_roads(self) -> List[str]:
        return {
            SignalPhase.NS_GO: ['N', 'S'],
            SignalPhase.EW_GO: ['E', 'W'],
            SignalPhase.ALL_RED: []
        }[self]

    @property
    def red_roads(self) -> List[str]:
        all_roads = ['N', 'S', 'E', 'W']
        return [r for r in all_roads if r not in self.green_roads]


@dataclass
class IntersectionState:
    """Snapshot of intersection state for AI"""
    timestamp: float
    phase: SignalPhase
    elapsed_green: float
    queues: Dict[str, int]
    waiting_times: Dict[str, float]
    arrival_rates: Dict[str, float]
    vehicles_served: int
    total_queue: int
    max_waiting_time: float
    avg_waiting_time: float

    def to_features(self) -> List[float]:
        """Convert state to normalized feature vector for AI"""
        return [
            self.queues.get('N', 0) / 50.0,
            self.queues.get('S', 0) / 50.0,
            self.queues.get('E', 0) / 50.0,
            self.queues.get('W', 0) / 50.0,
            self.arrival_rates.get('N', 0) / 10.0,
            self.arrival_rates.get('S', 0) / 10.0,
            self.arrival_rates.get('E', 0) / 10.0,
            self.arrival_rates.get('W', 0) / 10.0,
            1.0 if self.phase == SignalPhase.NS_GO else 0.0,
            self.elapsed_green / 60.0,
            self.avg_waiting_time / 60.0,
            self.max_waiting_time / 60.0,
        ]


class Intersection:
    """
    4-way intersection with traffic signal control.

    IMPORTANT: current_time passed to methods must be simulation time
    (starts at 0, increments by 1 each step). NEVER use time.time().
    """

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 40.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 1.0
    ):
        # Initialize roads
        self.roads: Dict[str, Road] = {
            'N': Road('N', saturation_flow=2.5),
            'S': Road('S', saturation_flow=2.5),
            'E': Road('E', saturation_flow=2.5),
            'W': Road('W', saturation_flow=2.5)
        }

        # Signal configuration
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration

        # Current state — ALL times are SIMULATION time
        self.phase = SignalPhase.NS_GO
        self.phase_start_time = 0.0   # <-- SIMULATION TIME, not wall clock
        self.is_transitioning = False

        # Statistics
        self.vehicles_served_total = 0
        self.total_waiting_time = 0.0
        self.phase_changes = 0
        self.history = []
        self._last_update_time = 0.0

        # Arrival rate tracking
        self._arrival_counts = {r: 0 for r in ['N', 'S', 'E', 'W']}
        self._arrival_window = 10
        self._arrival_history: List[Tuple[float, Dict[str, int]]] = []

    # ------------------------------------------------------------
    def add_vehicle(self, road_name: str, vehicle: Vehicle) -> bool:
        if road_name not in self.roads:
            return False
        success = self.roads[road_name].add_vehicle(vehicle)
        if success:
            self._arrival_counts[road_name] += 1
        return success

    # ------------------------------------------------------------
    def step(self, current_time: float, dt: float = 1.0):
        """Advance simulation by one step. current_time MUST be sim time."""
        # Update waiting times
        for road in self.roads.values():
            road.update_waiting_times(current_time)

        # Process departures
        self._process_departures(current_time, dt)

        # Update arrival rates
        self._update_arrival_rates(current_time)

        # Record state
        if int(current_time) % 5 == 0:
            self._record_state(current_time)

    def _process_departures(self, current_time: float, dt: float):
        green_roads = self.phase.green_roads
        for road_name in green_roads:
            road = self.roads[road_name]
            capacity = road.saturation_flow * dt
            departed = 0
            while departed < capacity and len(road) > 0:
                vehicle = road.remove_vehicle(current_time)
                if vehicle:
                    departed += 1
                    self.vehicles_served_total += 1
                    self.total_waiting_time += vehicle.waiting_time

    def _update_arrival_rates(self, current_time: float):
        self._arrival_history.append((current_time, self._arrival_counts.copy()))
        cutoff = current_time - self._arrival_window
        self._arrival_history = [
            (t, c) for t, c in self._arrival_history if t >= cutoff
        ]
        self._arrival_counts = {r: 0 for r in ['N', 'S', 'E', 'W']}

    def get_arrival_rates(self) -> Dict[str, float]:
        if len(self._arrival_history) < 2:
            return {r: 0.0 for r in ['N', 'S', 'E', 'W']}
        rates = {}
        for road in ['N', 'S', 'E', 'W']:
            total = sum(c[road] for _, c in self._arrival_history)
            rates[road] = total / max(1, self._arrival_window)
        return rates

    # ------------------------------------------------------------
    def switch_phase(self, new_phase: SignalPhase, current_time: float):
        if new_phase == self.phase:
            return
        elapsed = current_time - self.phase_start_time
        self.history.append({
            'timestamp': current_time,
            'old_phase': self.phase.value,
            'new_phase': new_phase.value,
            'elapsed': elapsed
        })
        self.phase = new_phase
        self.phase_start_time = current_time   # <-- SIM time
        self.phase_changes += 1

    def get_elapsed_green(self, current_time: float) -> float:
        """Elapsed time in current phase, in SIMULATION seconds."""
        return max(0.0, current_time - self.phase_start_time)

    def should_switch(self, current_time: float) -> bool:
        elapsed = self.get_elapsed_green(current_time)
        if elapsed < self.min_green:
            return False
        if elapsed >= self.max_green:
            return True
        return False

    # ------------------------------------------------------------
    def get_demand(self, road_name: str) -> float:
        if road_name in self.roads:
            return self.roads[road_name].get_demand()
        return 0.0

    def get_phase_demand(self, phase: SignalPhase) -> float:
        return sum(self.get_demand(r) for r in phase.green_roads)

    def get_queue_lengths(self) -> Dict[str, int]:
        return {n: r.get_queue_length() for n, r in self.roads.items()}

    def get_waiting_times(self) -> Dict[str, float]:
        return {
            n: r.get_waiting_time_summary()['avg']
            for n, r in self.roads.items()
        }

    def get_max_waiting_time(self) -> float:
        return max(
            (r.get_waiting_time_summary()['max'] for r in self.roads.values()),
            default=0.0
        )

    def get_average_waiting_time(self) -> float:
        if self.vehicles_served_total == 0:
            return 0.0
        return self.total_waiting_time / self.vehicles_served_total

    # ------------------------------------------------------------
    def get_state(self, current_time: float) -> IntersectionState:
        return IntersectionState(
            timestamp=current_time,
            phase=self.phase,
            elapsed_green=self.get_elapsed_green(current_time),
            queues=self.get_queue_lengths(),
            waiting_times=self.get_waiting_times(),
            arrival_rates=self.get_arrival_rates(),
            vehicles_served=self.vehicles_served_total,
            total_queue=sum(self.get_queue_lengths().values()),
            max_waiting_time=self.get_max_waiting_time(),
            avg_waiting_time=self.get_average_waiting_time()
        )

    def _record_state(self, current_time: float):
        state = self.get_state(current_time)
        self.history.append({
            'timestamp': current_time,
            'phase': self.phase.value,
            'queues': state.queues.copy(),
            'vehicles_served': state.vehicles_served,
            'avg_wait': state.avg_waiting_time
        })

    def get_total_vehicles(self) -> int:
        return sum(len(r) for r in self.roads.values())

    def get_throughput(self, window: float = 60.0) -> float:
        if len(self.history) < 2:
            return 0.0
        # Simple estimate based on total served over simulation time
        return 0.0   # placeholder — simulator computes this properly

    def get_statistics(self) -> Dict:
        return {
            'vehicles_served': self.vehicles_served_total,
            'average_wait_time': self.get_average_waiting_time(),
            'phase_changes': self.phase_changes,
            'queue_lengths': self.get_queue_lengths(),
            'total_queue': sum(self.get_queue_lengths().values()),
            'arrival_rates': self.get_arrival_rates(),
            'current_phase': self.phase.value,
        }

    def reset(self):
        for road in self.roads.values():
            road.clear()
        self.phase = SignalPhase.NS_GO
        self.phase_start_time = 0.0     # <-- reset to sim time 0
        self.vehicles_served_total = 0
        self.total_waiting_time = 0.0
        self.phase_changes = 0
        self.history = []
        self._arrival_counts = {r: 0 for r in ['N', 'S', 'E', 'W']}
        self._arrival_history = []
        self._last_update_time = 0.0

    def __repr__(self) -> str:
        q = self.get_queue_lengths()
        return (f"Intersection(phase={self.phase.value}, "
                f"N:{q['N']},S:{q['S']},E:{q['E']},W:{q['W']}, "
                f"served={self.vehicles_served_total})")