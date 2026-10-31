"""
Dataset Generation - Creates labeled training data for the Deep Learning model

FIXED VERSION:
- Uses SIMULATION time (not wall-clock)
- Properly cycles phases during data collection
- Ensures both NS_GO and EW_GO appear in the dataset
- Labels are meaningful (chosen by look-ahead cost evaluation)
"""

import numpy as np
import pandas as pd
import random
import os
from typing import List

from ..simulation.intersection import Intersection, SignalPhase
from ..simulation.traffic_generator import TrafficGenerator, TrafficScenario


class DatasetGenerator:
    """
    Generates labeled training dataset for traffic signal control.
    """

    FEATURE_NAMES = [
        'n_queue', 's_queue', 'e_queue', 'w_queue',
        'n_arrival_rate', 's_arrival_rate', 'e_arrival_rate', 'w_arrival_rate',
        'phase', 'elapsed_green',
        'avg_wait_time', 'max_wait_time', 'total_queue',
    ]
    TARGET_NAME = 'action'

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.samples: List[List[float]] = []

    # ------------------------------------------------------------------
    def get_state_features(self, intersection: Intersection,
                           current_time: float) -> np.ndarray:
        """Extract 13 features using SIMULATION time."""
        q = intersection.get_queue_lengths()
        a = intersection.get_arrival_rates()

        phase_val = 1.0 if intersection.phase == SignalPhase.NS_GO else 0.0
        elapsed = intersection.get_elapsed_green(current_time)  # sim-time-based

        features = np.array([
            q.get('N', 0) / 50.0,
            q.get('S', 0) / 50.0,
            q.get('E', 0) / 50.0,
            q.get('W', 0) / 50.0,
            a.get('N', 0) / 10.0,
            a.get('S', 0) / 10.0,
            a.get('E', 0) / 10.0,
            a.get('W', 0) / 10.0,
            phase_val,
            elapsed / 60.0,
            intersection.get_average_waiting_time() / 60.0,
            intersection.get_max_waiting_time() / 60.0,
            sum(q.values()) / 100.0,
        ], dtype=np.float32)

        return features

    # ------------------------------------------------------------------
    def evaluate_action(self, intersection: Intersection,
                        action: int, current_time: float,
                        horizon: int = 5) -> float:
        """
        Cost of each action:
          0 = KEEP current phase
          1 = SWITCH to other phase
          2 = EXTEND green by 5s
        Lower is better.
        """
        q = intersection.get_queue_lengths()
        total_q = sum(q.values())
        avg_wait = intersection.get_average_waiting_time()

        green_roads = intersection.phase.green_roads
        red_roads = [r for r in ['N', 'S', 'E', 'W'] if r not in green_roads]

        if action == 0:  # KEEP
            departures = sum(
                min(intersection.roads[r].saturation_flow * horizon,
                    intersection.roads[r].get_queue_length())
                for r in green_roads
            )
            arrivals = sum(
                intersection.get_arrival_rates().get(r, 0) * horizon
                for r in red_roads
            )
            new_q = total_q - departures + arrivals
            return new_q * 0.5 + avg_wait * 0.3

        elif action == 1:  # SWITCH
            # Switching clears current green; other phase gets served
            return total_q * 0.3 + avg_wait * 0.2 + 5.0  # switch penalty

        elif action == 2:  # EXTEND
            departures = sum(
                min(intersection.roads[r].saturation_flow * (horizon + 5),
                    intersection.roads[r].get_queue_length())
                for r in green_roads
            )
            arrivals = sum(
                intersection.get_arrival_rates().get(r, 0) * (horizon + 5)
                for r in ['N', 'S', 'E', 'W']
            )
            new_q = total_q - departures + arrivals
            return new_q * 0.4 + avg_wait * 0.2 + 2.0  # extend penalty

        return 0.0

    def get_optimal_action(self, intersection: Intersection,
                           current_time: float, horizon: int = 5) -> int:
        costs = [self.evaluate_action(intersection, a, current_time, horizon)
                 for a in [0, 1, 2]]
        return int(np.argmin(costs))

    # ------------------------------------------------------------------
    def generate_samples(
        self,
        num_samples: int = 5000,
        scenarios: List[TrafficScenario] = None,
        duration_per_scenario: float = 200.0,
        sampling_interval: float = 5.0,
        save_path: str = "dataset/traffic_dataset.csv"
    ) -> pd.DataFrame:

        if scenarios is None:
            scenarios = [
                TrafficScenario.NORMAL,
                TrafficScenario.MORNING_RUSH,
                TrafficScenario.EVENING_RUSH,
                TrafficScenario.BALANCED,
                TrafficScenario.HEAVY_NS,
                TrafficScenario.HEAVY_EW,
                TrafficScenario.SUDDEN_SPIKE,
                TrafficScenario.LOW_TRAFFIC,
            ]

        print("=" * 60)
        print("🚦 GENERATING CLEAN TRAINING DATASET")
        print("=" * 60)
        print(f"  Target:      {num_samples} samples")
        print(f"  Scenarios:   {len(scenarios)}")
        print(f"  Sample every {sampling_interval}s of simulation time")

        samples = []
        samples_per_scenario = num_samples // len(scenarios)

        for scenario in scenarios:
            print(f"\n  ▸ {scenario.value}  (target {samples_per_scenario})")
            collected = 0
            run_idx = 0

            while collected < samples_per_scenario and run_idx < 30:
                run_idx += 1
                random.seed(self.seed + run_idx)
                np.random.seed(self.seed + run_idx)

                intersection = Intersection(min_green=8.0, max_green=30.0)
                generator = TrafficGenerator(seed=self.seed + run_idx)
                generator.set_scenario(scenario)
                generator.apply_scenario_to_intersection(intersection)

                # Force phase cycling so BOTH phases appear in data
                # We deliberately switch every 12 seconds
                forced_phase_switch_interval = 12.0

                t = 0.0
                next_sample_t = 0.0
                next_switch_t = forced_phase_switch_interval

                while t < duration_per_scenario and collected < samples_per_scenario:
                    # Generate vehicles
                    new_v = generator.generate_vehicles(t, 1.0)
                    for road, vehs in new_v.items():
                        for v in vehs:
                            intersection.add_vehicle(road, v)

                    # FORCE phase switch every 12s so both phases appear
                    if t >= next_switch_t:
                        new_phase = (SignalPhase.EW_GO
                                     if intersection.phase == SignalPhase.NS_GO
                                     else SignalPhase.NS_GO)
                        intersection.switch_phase(new_phase, t)
                        next_switch_t += forced_phase_switch_interval

                    # Step intersection
                    intersection.step(t, 1.0)

                    # Sample
                    if t >= next_sample_t:
                        feats = self.get_state_features(intersection, t)
                        action = self.get_optimal_action(intersection, t)
                        samples.append(list(feats) + [action])
                        collected += 1
                        next_sample_t += sampling_interval

                    t += 1.0

                print(f"      run {run_idx}: collected {collected}/{samples_per_scenario}")

        df = pd.DataFrame(samples, columns=self.FEATURE_NAMES + [self.TARGET_NAME])

        print("\n" + "=" * 60)
        print(f"✅ Generated {len(df)} samples")
        print(f"   Feature 8 (phase) distribution:")
        print(f"     NS_GO (=1.0): {(df['phase']==1.0).sum()}")
        print(f"     EW_GO (=0.0): {(df['phase']==0.0).sum()}")
        print(f"   Feature 9 (elapsed_green) stats:")
        print(f"     min={df['elapsed_green'].min():.4f}  "
              f"max={df['elapsed_green'].max():.4f}  "
              f"mean={df['elapsed_green'].mean():.4f}")
        print(f"   Action distribution:")
        print(df['action'].value_counts().to_string())
        print("=" * 60)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved to {save_path}")

        return df