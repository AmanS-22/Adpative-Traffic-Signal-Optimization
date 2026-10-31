"""
Unit tests for Intersection class
Run with: python -m pytest tests/test_intersection.py -v
"""

import pytest
import time
from src.simulation.intersection import Intersection, SignalPhase
from src.simulation.vehicle import Vehicle, VehicleType


class TestIntersection:
    """Test Intersection class"""
    
    def test_intersection_creation(self):
        """Test creating an intersection"""
        intersection = Intersection()
        
        assert len(intersection.roads) == 4
        assert 'N' in intersection.roads
        assert 'S' in intersection.roads
        assert 'E' in intersection.roads
        assert 'W' in intersection.roads
        assert intersection.phase == SignalPhase.NS_GO
        assert intersection.min_green == 10.0
        assert intersection.max_green == 40.0
    
    def test_add_vehicle(self):
        """Test adding vehicles to intersection"""
        intersection = Intersection()
        
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'N')
        result = intersection.add_vehicle('N', vehicle)
        
        assert result == True
        assert intersection.roads['N'].get_queue_length() == 1
        assert intersection.get_total_vehicles() == 1
    
    def test_add_vehicle_invalid_road(self):
        """Test adding vehicle to invalid road"""
        intersection = Intersection()
        
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'X')
        result = intersection.add_vehicle('X', vehicle)
        
        assert result == False
        assert intersection.get_total_vehicles() == 0
    
    def test_step(self):
        """Test stepping the simulation"""
        intersection = Intersection()
        
        # Add vehicle to North road (green in NS_GO)
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'N')
        intersection.add_vehicle('N', vehicle)
        
        # Step forward - vehicle should depart
        current_time = 1.0
        intersection.step(current_time, dt=1.0)
        
        # Vehicle should have departed (N road has green in NS_GO)
        assert intersection.roads['N'].get_queue_length() == 0
        assert intersection.vehicles_served_total == 1
    
    def test_step_no_departure_red(self):
        """Test that vehicles don't depart on red"""
        intersection = Intersection()
        intersection.phase = SignalPhase.NS_GO
        
        # Add vehicle to East road (red in NS_GO)
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'E')
        intersection.add_vehicle('E', vehicle)
        
        # Step forward
        current_time = 1.0
        intersection.step(current_time, dt=1.0)
        
        # Vehicle should still be waiting (E road has red)
        assert intersection.roads['E'].get_queue_length() == 1
        assert intersection.vehicles_served_total == 0
    
    def test_switch_phase(self):
        """Test phase switching"""
        intersection = Intersection()
        current_time = time.time()
        
        assert intersection.phase == SignalPhase.NS_GO
        
        intersection.switch_phase(SignalPhase.EW_GO, current_time)
        
        assert intersection.phase == SignalPhase.EW_GO
        assert intersection.phase_changes == 1
    
    def test_should_switch(self):
        """Test should_switch logic"""
        intersection = Intersection(min_green=5.0, max_green=15.0)
        intersection.phase_start_time = time.time() - 3.0  # 3 seconds elapsed
        
        current_time = time.time()
        
        # Should not switch (min green not reached)
        assert intersection.should_switch(current_time) == False
        
        # Fast forward to 5 seconds (min green)
        intersection.phase_start_time = time.time() - 5.0
        assert intersection.should_switch(time.time()) == False
        
        # Fast forward to 15 seconds (max green)
        intersection.phase_start_time = time.time() - 15.0
        assert intersection.should_switch(time.time()) == True
    
    def test_get_phase_demand(self):
        """Test phase demand calculation"""
        intersection = Intersection()
        
        # Add vehicles to different roads
        for road in ['N', 'S']:  # NS_GO phase
            for i in range(5):
                v = Vehicle(VehicleType.CAR, float(i), road)
                intersection.add_vehicle(road, v)
        
        for road in ['E', 'W']:  # EW_GO phase
            for i in range(3):
                v = Vehicle(VehicleType.CAR, float(i), road)
                intersection.add_vehicle(road, v)
        
        ns_demand = intersection.get_phase_demand(SignalPhase.NS_GO)
        ew_demand = intersection.get_phase_demand(SignalPhase.EW_GO)
        
        # NS should have higher demand (10 vehicles vs 6)
        assert ns_demand > ew_demand
    
    def test_get_state(self):
        """Test getting intersection state"""
        intersection = Intersection()
        
        # Add some vehicles
        for road in ['N', 'S']:
            v = Vehicle(VehicleType.CAR, 0.0, road)
            intersection.add_vehicle(road, v)
        
        current_time = time.time()
        state = intersection.get_state(current_time)
        
        assert state.phase == SignalPhase.NS_GO
        assert state.queues['N'] == 1
        assert state.queues['S'] == 1
        assert state.elapsed_green >= 0
        assert state.vehicles_served == 0
    
    def test_get_statistics(self):
        """Test getting statistics"""
        intersection = Intersection()
        
        # Add and process some vehicles
        for i in range(5):
            v = Vehicle(VehicleType.CAR, float(i), 'N')
            intersection.add_vehicle('N', v)
        
        for t in range(1, 6):
            intersection.step(float(t), dt=1.0)
        
        stats = intersection.get_statistics()
        
        assert 'vehicles_served' in stats
        assert 'average_wait_time' in stats
        assert 'phase_changes' in stats
        assert 'queue_lengths' in stats
        assert 'arrival_rates' in stats
    
    def test_reset(self):
        """Test resetting intersection"""
        intersection = Intersection()
        
        # Add vehicles
        v = Vehicle(VehicleType.CAR, 0.0, 'N')
        intersection.add_vehicle('N', v)
        
        # Switch phase
        intersection.switch_phase(SignalPhase.EW_GO, time.time())
        
        assert intersection.get_total_vehicles() == 1
        assert intersection.phase_changes == 1
        
        # Reset
        intersection.reset()
        
        assert intersection.get_total_vehicles() == 0
        assert intersection.phase_changes == 0
        assert intersection.phase == SignalPhase.NS_GO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])