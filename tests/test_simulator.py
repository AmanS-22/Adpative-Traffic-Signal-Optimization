"""
Unit tests for Simulator
Run with: python -m pytest tests/test_simulator.py -v
"""

import pytest
import time
from src.simulation.simulator import Simulator, SimulationConfig, ControllerType
from src.simulation.traffic_generator import TrafficScenario


class TestSimulator:
    """Test Simulator class"""
    
    def test_simulator_creation(self):
        """Test creating a simulator"""
        config = SimulationConfig(duration=10.0)
        simulator = Simulator(config)
        
        assert simulator is not None
        assert simulator.config.duration == 10.0
        assert simulator.intersection is not None
        assert simulator.generator is not None
    
    def test_run_fixed_controller(self):
        """Test running with fixed controller"""
        config = SimulationConfig(
            duration=10.0,
            controller=ControllerType.FIXED,
            fixed_green_time=5.0,
            seed=42
        )
        simulator = Simulator(config)
        result = simulator.run()
        
        assert result is not None
        assert result.vehicles_served >= 0
        assert result.average_waiting_time >= 0
        assert result.phase_changes > 0
    
    def test_run_rule_based_controller(self):
        """Test running with rule-based controller"""
        config = SimulationConfig(
            duration=10.0,
            controller=ControllerType.RULE_BASED,
            seed=42
        )
        simulator = Simulator(config)
        result = simulator.run()
        
        assert result is not None
        assert result.vehicles_served >= 0
        assert result.average_waiting_time >= 0
    
    def test_run_with_scenarios(self):
        """Test running with different scenarios"""
        for scenario in [TrafficScenario.NORMAL, TrafficScenario.HEAVY_NS]:
            config = SimulationConfig(
                duration=10.0,
                scenario=scenario,
                controller=ControllerType.FIXED,
                seed=42
            )
            simulator = Simulator(config)
            result = simulator.run()
            
            assert result is not None
            assert result.config.scenario == scenario
    
    def test_multiple_runs(self):
        """Test running multiple simulations"""
        config = SimulationConfig(
            duration=5.0,
            controller=ControllerType.FIXED,
            seed=42
        )
        simulator = Simulator(config)
        results = simulator.run_multiple(num_runs=3)
        
        assert len(results) == 3
        for result in results:
            assert result.vehicles_served >= 0
    
    def test_controller_comparison(self):
        """Test controller comparison"""
        config = SimulationConfig(duration=10.0)
        simulator = Simulator(config)
        
        scenarios = [TrafficScenario.NORMAL, TrafficScenario.HEAVY_NS]
        results = simulator.compare_controllers(
            scenarios=scenarios,
            num_runs=2,
            duration=10.0
        )
        
        assert len(results) == 2  # Fixed and Rule-based
        assert 'fixed' in results
        assert 'rule_based' in results
        assert 'normal' in results['fixed']
        assert 'heavy_ns' in results['fixed']
    
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        config1 = SimulationConfig(duration=10.0, controller=ControllerType.FIXED, seed=42)
        config2 = SimulationConfig(duration=10.0, controller=ControllerType.FIXED, seed=42)
        
        simulator1 = Simulator(config1)
        simulator2 = Simulator(config2)
        
        result1 = simulator1.run()
        result2 = simulator2.run()
        
        assert result1.vehicles_served == result2.vehicles_served
        assert result1.average_waiting_time == result2.average_waiting_time
    
    def test_print_results(self):
        """Test printing results (just ensure it doesn't crash)"""
        config = SimulationConfig(duration=5.0, controller=ControllerType.FIXED)
        simulator = Simulator(config)
        result = simulator.run()
        
        # Should not raise any exceptions
        simulator.print_results(result)
    
    def test_statistics_accuracy(self):
        """Test that statistics are collected correctly"""
        config = SimulationConfig(
            duration=20.0,
            controller=ControllerType.FIXED,
            fixed_green_time=5.0,
            seed=42,
            verbose=False
        )
        simulator = Simulator(config)
        result = simulator.run()
        
        # Basic sanity checks
        assert result.vehicles_served >= 0
        assert result.average_waiting_time >= 0
        assert result.max_queue_length >= 0
        assert result.throughput >= 0
        assert result.signal_utilization >= 0
        assert result.congestion_percentage >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])