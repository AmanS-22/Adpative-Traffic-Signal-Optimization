"""
Unit tests for Vehicle and Road classes
Run with: python -m pytest tests/test_vehicle_road.py -v
"""

import pytest
import time
from src.simulation.vehicle import Vehicle, VehicleType
from src.simulation.road import Road


class TestVehicle:
    """Test Vehicle class"""
    
    def test_vehicle_creation(self):
        """Test creating a vehicle"""
        vehicle = Vehicle(
            vehicle_type=VehicleType.CAR,
            arrival_time=10.0,
            road='N'
        )
        
        assert vehicle.vehicle_type == VehicleType.CAR
        assert vehicle.arrival_time == 10.0
        assert vehicle.road == 'N'
        assert vehicle.id is not None
        assert vehicle.waiting_time == 0.0
        assert vehicle.departure_time is None
        assert vehicle.priority == 1
    
    def test_vehicle_types(self):
        """Test different vehicle types"""
        car = Vehicle(VehicleType.CAR, 1.0, 'N')
        bike = Vehicle(VehicleType.BIKE, 1.0, 'N')
        bus = Vehicle(VehicleType.BUS, 1.0, 'N')
        truck = Vehicle(VehicleType.TRUCK, 1.0, 'N')
        
        assert car.vehicle_type.length == 4.5
        assert bike.vehicle_type.length == 2.0
        assert bus.vehicle_type.length == 12.0
        assert truck.vehicle_type.length == 15.0
        
        assert car.vehicle_type.saturation_factor == 1.0
        assert bus.vehicle_type.saturation_factor == 2.5
    
    def test_random_vehicle_type(self):
        """Test random vehicle type generation"""
        types = [VehicleType.random_type() for _ in range(100)]
        
        # Should only contain valid types
        for t in types:
            assert t in VehicleType
        
        # Should have more cars than other types (statistically)
        car_count = sum(1 for t in types if t == VehicleType.CAR)
        assert car_count > 50  # ~70% should be cars
    
    def test_vehicle_update_waiting(self):
        """Test updating waiting time with time differences"""
        vehicle = Vehicle(VehicleType.CAR, 10.0, 'N')
        
        assert vehicle.waiting_time == 0.0
        
        # Update after 5 seconds
        vehicle.update_waiting_time(15.0)
        assert vehicle.waiting_time == 5.0
        
        # Update after another 5 seconds (total 10)
        vehicle.update_waiting_time(20.0)
        assert vehicle.waiting_time == 10.0
    
    def test_vehicle_update_waiting_multiple_calls(self):
        """Test multiple waiting time updates"""
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'N')
        
        vehicle.update_waiting_time(2.0)   # Should wait 2 seconds
        assert vehicle.waiting_time == 2.0
        
        vehicle.update_waiting_time(3.5)   # Should wait 1.5 more seconds
        assert vehicle.waiting_time == 3.5
        
        vehicle.update_waiting_time(5.0)   # Should wait 1.5 more seconds
        assert vehicle.waiting_time == 5.0
    
    def test_vehicle_depart(self):
        """Test vehicle departure"""
        vehicle = Vehicle(VehicleType.CAR, 10.0, 'N')
        
        assert vehicle.departure_time is None
        assert vehicle.travel_time == 0.0
        
        vehicle.depart(15.0)
        
        assert vehicle.departure_time == 15.0
        assert vehicle.travel_time == 5.0
        assert not vehicle.is_waiting
    
    def test_vehicle_to_dict(self):
        """Test vehicle to dictionary conversion"""
        vehicle = Vehicle(VehicleType.CAR, 10.0, 'N')
        vehicle.update_waiting_time(15.0)  # Wait 5 seconds
        
        vehicle_dict = vehicle.to_dict()
        
        assert vehicle_dict['id'] == vehicle.id
        assert vehicle_dict['type'] == 'car'
        assert vehicle_dict['arrival_time'] == 10.0
        assert vehicle_dict['road'] == 'N'
        assert vehicle_dict['waiting_time'] == 5.0
        assert 'attributes' in vehicle_dict


class TestRoad:
    """Test Road class"""
    
    def test_road_creation(self):
        """Test creating a road"""
        road = Road('N', max_length=50, saturation_flow=3.0)
        
        assert road.name == 'N'
        assert road.max_length == 50
        assert road.saturation_flow == 3.0
        assert road.get_queue_length() == 0
        assert road.vehicles_served == 0
    
    def test_add_vehicle(self):
        """Test adding vehicles to road"""
        road = Road('N')
        
        vehicle1 = Vehicle(VehicleType.CAR, 1.0, 'N')
        vehicle2 = Vehicle(VehicleType.BUS, 2.0, 'N')
        
        assert road.add_vehicle(vehicle1) == True
        assert road.get_queue_length() == 1
        
        assert road.add_vehicle(vehicle2) == True
        assert road.get_queue_length() == 2
    
    def test_remove_vehicle(self):
        """Test removing vehicles from road"""
        road = Road('N')
        
        vehicle = Vehicle(VehicleType.CAR, 1.0, 'N')
        road.add_vehicle(vehicle)
        
        removed = road.remove_vehicle(10.0)
        
        assert removed is not None
        assert removed.departure_time == 10.0
        assert road.get_queue_length() == 0
        assert road.vehicles_served == 1
    
    def test_queue_fifo(self):
        """Test FIFO order of queue"""
        road = Road('N')
        
        v1 = Vehicle(VehicleType.CAR, 1.0, 'N')
        v2 = Vehicle(VehicleType.BUS, 2.0, 'N')
        v3 = Vehicle(VehicleType.CAR, 3.0, 'N')
        
        road.add_vehicle(v1)
        road.add_vehicle(v2)
        road.add_vehicle(v3)
        
        removed1 = road.remove_vehicle(10.0)
        removed2 = road.remove_vehicle(11.0)
        removed3 = road.remove_vehicle(12.0)
        
        assert removed1 == v1
        assert removed2 == v2
        assert removed3 == v3
    
    def test_max_queue_limit(self):
        """Test maximum queue length limit"""
        road = Road('N', max_length=2)
        
        v1 = Vehicle(VehicleType.CAR, 1.0, 'N')
        v2 = Vehicle(VehicleType.CAR, 2.0, 'N')
        v3 = Vehicle(VehicleType.CAR, 3.0, 'N')
        
        assert road.add_vehicle(v1) == True
        assert road.add_vehicle(v2) == True
        assert road.add_vehicle(v3) == False  # Should fail, queue full
        
        assert road.get_queue_length() == 2
    
    def test_waiting_times_fixed(self):
        """Test waiting time tracking - FIXED"""
        road = Road('N')
        
        # Add vehicle at time 0
        vehicle = Vehicle(VehicleType.CAR, 0.0, 'N')
        road.add_vehicle(vehicle)
        
        # Update waiting times at different times
        road.update_waiting_times(5.0)   # Wait 5 seconds
        road.update_waiting_times(10.0)  # Wait 5 more seconds
        
        # Vehicle should have waited 10 seconds total
        assert vehicle.waiting_time == 10.0
    
    def test_get_demand(self):
        """Test demand calculation"""
        road = Road('N')
        
        # Empty road - demand should be 0
        assert road.get_demand() == 0.0
        
        # Add 30 vehicles - demand should be 1.0 (full)
        for i in range(30):
            v = Vehicle(VehicleType.CAR, float(i), 'N')
            road.add_vehicle(v)
        
        assert road.get_demand() == 1.0
    
    def test_weighted_queue_length(self):
        """Test weighted queue length"""
        road = Road('N')
        
        # Add different vehicle types
        car = Vehicle(VehicleType.CAR, 1.0, 'N')
        bus = Vehicle(VehicleType.BUS, 2.0, 'N')
        truck = Vehicle(VehicleType.TRUCK, 3.0, 'N')
        
        road.add_vehicle(car)
        road.add_vehicle(bus)
        road.add_vehicle(truck)
        
        # Weighted sum: car(1.0) + bus(2.5) + truck(3.0) = 6.5
        assert road.get_weighted_queue_length() == 6.5
    
    def test_vehicle_type_counts(self):
        """Test vehicle type counting"""
        road = Road('N')
        
        car1 = Vehicle(VehicleType.CAR, 1.0, 'N')
        car2 = Vehicle(VehicleType.CAR, 2.0, 'N')
        bus = Vehicle(VehicleType.BUS, 3.0, 'N')
        
        road.add_vehicle(car1)
        road.add_vehicle(car2)
        road.add_vehicle(bus)
        
        types = road.get_vehicle_types()
        
        assert types['car'] == 2
        assert types['bus'] == 1
        assert types['bike'] == 0
        assert types['truck'] == 0
    
    def test_waiting_time_summary_fixed(self):
        """Test waiting time summary - FIXED"""
        road = Road('N')
        
        # Add vehicles with different arrival times
        v1 = Vehicle(VehicleType.CAR, 0.0, 'N')
        v2 = Vehicle(VehicleType.CAR, 2.0, 'N')
        v3 = Vehicle(VehicleType.CAR, 5.0, 'N')
        
        road.add_vehicle(v1)
        road.add_vehicle(v2)
        road.add_vehicle(v3)
        
        # Update at time 10
        road.update_waiting_times(10.0)
        
        summary = road.get_waiting_time_summary()
        
        # v3 arrived at 5, waited 5 seconds (10 - 5 = 5)
        # v2 arrived at 2, waited 8 seconds (10 - 2 = 8)
        # v1 arrived at 0, waited 10 seconds (10 - 0 = 10)
        assert summary['min'] == 5.0
        assert summary['max'] == 10.0
        assert summary['avg'] == (10.0 + 8.0 + 5.0) / 3
    
    def test_road_to_dict(self):
        """Test road to dictionary conversion"""
        road = Road('N')
        
        v1 = Vehicle(VehicleType.CAR, 1.0, 'N')
        v2 = Vehicle(VehicleType.BUS, 2.0, 'N')
        
        road.add_vehicle(v1)
        road.add_vehicle(v2)
        road.update_waiting_times(5.0)
        
        road_dict = road.to_dict()
        
        assert road_dict['name'] == 'N'
        assert road_dict['queue_length'] == 2
        assert 'waiting_time_summary' in road_dict
        assert 'vehicle_types' in road_dict
        assert road_dict['vehicle_types']['car'] == 1
        assert road_dict['vehicle_types']['bus'] == 1
    
    def test_max_queue_observed(self):
        """Test max queue length tracking"""
        road = Road('N')
        
        # Add 5 vehicles
        for i in range(5):
            v = Vehicle(VehicleType.CAR, float(i), 'N')
            road.add_vehicle(v)
        
        assert road.max_queue_length_observed == 5
        
        # Remove some
        for i in range(3):
            road.remove_vehicle(float(i + 5))
        
        assert road.max_queue_length_observed == 5  # Should still be 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])