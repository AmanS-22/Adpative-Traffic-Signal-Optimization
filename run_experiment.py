"""
Run complete experiment comparing all controllers.

KEY PRINCIPLE: Fair comparison.
For each (scenario, seed), ONE vehicle schedule is pre-generated and
fed to ALL controllers. This guarantees identical traffic input.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulation.simulator import Simulator, SimulationConfig, ControllerType
from src.simulation.traffic_generator import TrafficGenerator, TrafficScenario


SEEDS = [42, 123, 456]      # Multi-seed trials
DURATION = 120.0            # seconds per simulation
DT = 1.0


def run_complete_experiment():
    print("=" * 72)
    print("🚦 CONTROLLER COMPARISON — FAIR TRAFFIC, MULTI-SEED")
    print("=" * 72)

    os.makedirs('results', exist_ok=True)

    scenarios = [
        TrafficScenario.NORMAL,
        TrafficScenario.MORNING_RUSH,
        TrafficScenario.EVENING_RUSH,
        TrafficScenario.HEAVY_NS,
        TrafficScenario.HEAVY_EW,
        TrafficScenario.SUDDEN_SPIKE,
        TrafficScenario.LOW_TRAFFIC,
    ]
    controllers = [
        ControllerType.FIXED,
        ControllerType.RULE_BASED,
        ControllerType.DEEP_LEARNING,
    ]

    rows = []

    for scenario in scenarios:
        print(f"\n{'=' * 72}")
        print(f"🎭 Scenario: {scenario.value}")
        print(f"{'=' * 72}")

        # === CRITICAL: ONE schedule per (scenario, seed) ===
        # Reused for ALL controllers
        seed_schedules = {}
        for seed in SEEDS:
            gen = TrafficGenerator(seed=seed)
            gen.set_scenario(scenario)
            seed_schedules[seed] = gen.generate_schedule(
                duration=DURATION, dt=DT
            )

        for controller in controllers:
            print(f"\n📊 {controller.value.upper()}")

            per_seed = []
            for seed in SEEDS:
                cfg = SimulationConfig(
                    duration=DURATION,
                    dt=DT,
                    scenario=scenario,
                    controller=controller,
                    seed=seed,
                    fixed_green_time=20.0,
                    rule_min_green=10.0,
                    rule_max_green=40.0,
                    verbose=False,
                )
                sim = Simulator(cfg)
                result = sim.run(schedule=seed_schedules[seed])
                per_seed.append(result)

                print(f"   seed={seed}: served={result.vehicles_served}  "
                      f"wait={result.average_waiting_time:.2f}s  "
                      f"queue={result.average_queue_length:.2f}  "
                      f"thrpt={result.throughput:.0f}  "
                      f"phases={result.phase_changes}")

            # Aggregate
            agg = {
                'scenario': scenario.value,
                'controller': controller.value,
                'vehicles_served_mean': float(np.mean([r.vehicles_served for r in per_seed])),
                'vehicles_served_std': float(np.std([r.vehicles_served for r in per_seed])),
                'avg_wait_mean': float(np.mean([r.average_waiting_time for r in per_seed])),
                'avg_wait_std': float(np.std([r.average_waiting_time for r in per_seed])),
                'avg_queue_mean': float(np.mean([r.average_queue_length for r in per_seed])),
                'avg_queue_std': float(np.std([r.average_queue_length for r in per_seed])),
                'max_queue_mean': float(np.mean([r.max_queue_length for r in per_seed])),
                'throughput_mean': float(np.mean([r.throughput for r in per_seed])),
                'throughput_std': float(np.std([r.throughput for r in per_seed])),
                'phase_changes_mean': float(np.mean([r.phase_changes for r in per_seed])),
            }
            rows.append(agg)

            print(f"   → mean: served={agg['vehicles_served_mean']:.1f}"
                  f"±{agg['vehicles_served_std']:.1f}  "
                  f"wait={agg['avg_wait_mean']:.2f}"
                  f"±{agg['avg_wait_std']:.2f}s")

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = 'results/experiment_results.csv'
    try:
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved {csv_path}")
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt = f'results/experiment_results_{ts}.csv'
        df.to_csv(alt, index=False)
        print(f"\n⚠️ Could not write {csv_path} (file locked). Saved {alt}")

    # Plot
    plot_comparison(df)
    return df


# ----------------------------------------------------------------------
def plot_comparison(df: pd.DataFrame):
    scenarios = df['scenario'].unique()
    controllers = df['controller'].unique()
    colors = {'fixed': '#e74c3c', 'rule_based': '#3498db',
              'deep_learning': '#2ecc71'}

    metrics = [
        ('vehicles_served_mean', 'Vehicles Served (higher=better)'),
        ('avg_wait_mean', 'Avg Waiting Time (lower=better)'),
        ('avg_queue_mean', 'Avg Queue Length (lower=better)'),
        ('max_queue_mean', 'Max Queue Length (lower=better)'),
        ('throughput_mean', 'Throughput veh/h (higher=better)'),
        ('phase_changes_mean', 'Phase Changes'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(len(scenarios))
    width = 0.25

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        for j, ctrl in enumerate(controllers):
            vals = [
                df[(df['scenario'] == s) & (df['controller'] == ctrl)][metric].values[0]
                for s in scenarios
            ]
            ax.bar(x + j * width, vals, width,
                   label=ctrl.replace('_', ' ').title(),
                   color=colors.get(ctrl, 'gray'), alpha=0.85)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios],
                           rotation=30, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    plt.suptitle('🚦 Controller Comparison (Mean over %d seeds)' % len(SEEDS),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved results/experiment_results.png")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = run_complete_experiment()
    print("\n🎉 Experiment complete. See results/ folder.")