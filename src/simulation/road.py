"""
Road Module - Represents a road approach with vehicle queue
"""

from typing import List, Optional, Dict, Any
from collections import deque
import copy
from .vehicle import Vehicle, VehicleType


class Road:
    """
    Represents a road approach to the intersection
    
    Attributes:
        name: Road name (N, S, E, W)
        queue: Queue of vehicles waiting
        max_length: Maximum road length
        saturation_flow: Maximum vehicles per second that can pass
        vehicles_served: Total vehicles served
        total_waiting_time: Cumulative waiting time of all vehicles
    """
    
    def __init__(
        self,
        name: str,
        max_length: int = 100,
        saturation_flow: float = 2.5
    ):
        self.name = name
        self.max_length = max_length
        self.saturation_flow = saturation_flow
        self.queue: deque = deque()
        self.vehicles_served = 0
        self.total_waiting_time = 0.0
        self.max_queue_length_observed = 0
        self.history = []
        self._last_update_time = 0.0
    
    def add_vehicle(self, vehicle: Vehicle) -> bool:
        """
        Add a vehicle to the road queue
        
        Args:
            vehicle: Vehicle to add
            
        Returns:
            bool: True if successfully added, False if road is full
        """
        if len(self.queue) >= self.max_length:
            return False
        
        self.queue.append(vehicle)
        self._update_history()
        return True
    
    def remove_vehicle(self, current_time: float) -> Optional[Vehicle]:
        """
        Remove a vehicle from the front of the queue (departure)
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Vehicle: Departed vehicle or None if queue is empty
        """
        if not self.queue:
            return None
        
        # Update waiting time before removal
        self.update_waiting_times(current_time)
        
        vehicle = self.queue.popleft()
        vehicle.depart(current_time)
        self.vehicles_served += 1
        self.total_waiting_time += vehicle.waiting_time
        
        self._update_history()
        return vehicle
    
    def get_queue_length(self) -> int:
        """Get current queue length"""
        return len(self.queue)
    
    def get_waiting_time_summary(self) -> Dict[str, float]:
        """Get waiting time statistics"""
        if not self.queue:
            return {'min': 0, 'max': 0, 'avg': 0, 'total': self.total_waiting_time}
        
        wait_times = [v.waiting_time for v in self.queue]
        return {
            'min': min(wait_times),
            'max': max(wait_times),
            'avg': sum(wait_times) / len(wait_times),
            'total': self.total_waiting_time
        }
    
    def get_vehicles(self) -> List[Vehicle]:
        """Get all vehicles in queue"""
        return list(self.queue)
    
    def get_vehicle_types(self) -> Dict[str, int]:
        """Get count of each vehicle type in queue"""
        type_counts = {vt.value: 0 for vt in VehicleType}
        for vehicle in self.queue:
            type_counts[vehicle.vehicle_type.value] += 1
        return type_counts
    
    def get_demand(self) -> float:
        """
        Calculate current demand based on queue and arrival patterns
        Returns a normalized demand value (0-1)
        """
        queue_len = self.get_queue_length()
        # Normalize to max length (100 vehicles)
        demand = min(1.0, queue_len / 30.0)  # 30 vehicles = full demand
        return demand
    
    def get_weighted_queue_length(self) -> float:
        """
        Calculate weighted queue length considering vehicle types
        Buses and trucks consume more capacity
        """
        total_weight = 0.0
        for vehicle in self.queue:
            total_weight += vehicle.vehicle_type.saturation_factor
        return total_weight
    
    def update_waiting_times(self, current_time: float):
        """
        Update waiting times for all vehicles in queue
        Uses time difference since last update
        """
        if current_time > self._last_update_time:
            for vehicle in self.queue:
                vehicle.update_waiting_time(current_time)
            self._last_update_time = current_time
    
    def clear(self):
        """Clear all vehicles from the road"""
        self.queue.clear()
        self._update_history()
    
    def _update_history(self):
        """Update queue length history"""
        self.history.append({
            'time': len(self.history),
            'queue_length': self.get_queue_length(),
            'vehicles_served': self.vehicles_served
        })
        
        # Update maximum observed queue length
        current_len = self.get_queue_length()
        if current_len > self.max_queue_length_observed:
            self.max_queue_length_observed = current_len
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert road state to dictionary"""
        return {
            'name': self.name,
            'queue_length': self.get_queue_length(),
            'vehicles_served': self.vehicles_served,
            'total_waiting_time': self.total_waiting_time,
            'max_queue_observed': self.max_queue_length_observed,
            'waiting_time_summary': self.get_waiting_time_summary(),
            'vehicle_types': self.get_vehicle_types(),
            'demand': self.get_demand()
        }
    
    def __len__(self) -> int:
        return self.get_queue_length()
    
    def __repr__(self) -> str:
        return f"Road(name={self.name}, queue_len={self.get_queue_length()}, served={self.vehicles_served})"