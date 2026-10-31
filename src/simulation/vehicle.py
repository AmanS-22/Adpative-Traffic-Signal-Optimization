"""
Vehicle Module - Represents individual vehicles in the traffic simulation
"""

import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import random


class VehicleType(Enum):
    """Vehicle types with different characteristics"""
    CAR = "car"
    BIKE = "bike"
    BUS = "bus"
    TRUCK = "truck"
    
    @property
    def length(self) -> float:
        """Vehicle length in meters"""
        return {
            VehicleType.CAR: 4.5,
            VehicleType.BIKE: 2.0,
            VehicleType.BUS: 12.0,
            VehicleType.TRUCK: 15.0
        }[self]
    
    @property
    def saturation_factor(self) -> float:
        """How much this vehicle type contributes to saturation"""
        return {
            VehicleType.CAR: 1.0,
            VehicleType.BIKE: 0.5,
            VehicleType.BUS: 2.5,
            VehicleType.TRUCK: 3.0
        }[self]
    
    @property
    def priority(self) -> int:
        """Priority for emergency/preemption (higher = more priority)"""
        return {
            VehicleType.CAR: 1,
            VehicleType.BIKE: 1,
            VehicleType.BUS: 2,  # Slight priority for public transport
            VehicleType.TRUCK: 1
        }[self]
    
    @property
    def color(self) -> str:
        """Color for visualization"""
        return {
            VehicleType.CAR: "#4A90D9",
            VehicleType.BIKE: "#E67E22",
            VehicleType.BUS: "#2ECC71",
            VehicleType.TRUCK: "#E74C3C"
        }[self]
    
    @staticmethod
    def random_type() -> 'VehicleType':
        """Generate random vehicle type with realistic distribution"""
        weights = [0.70, 0.15, 0.10, 0.05]  # Car, Bike, Bus, Truck
        return random.choices(
            [VehicleType.CAR, VehicleType.BIKE, VehicleType.BUS, VehicleType.TRUCK],
            weights=weights
        )[0]


@dataclass
class Vehicle:
    """
    Represents a vehicle in the traffic simulation
    
    Attributes:
        id: Unique identifier
        vehicle_type: Type of vehicle (car, bike, bus, truck)
        arrival_time: Time when vehicle arrived at intersection
        road: Current road (N, S, E, W)
        waiting_time: Total waiting time in seconds
        travel_time: Total travel time through intersection
        priority: Priority level for emergency/preemption
        attributes: Additional vehicle attributes
    """
    vehicle_type: VehicleType
    arrival_time: float
    road: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    waiting_time: float = 0.0
    travel_time: float = 0.0
    departure_time: Optional[float] = None
    priority: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)
    _last_update_time: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize priority from vehicle type"""
        self.priority = self.vehicle_type.priority
        self._last_update_time = self.arrival_time
    
    def update_waiting_time(self, current_time: float):
        """
        Update waiting time for this vehicle
        
        Args:
            current_time: Current simulation time
        """
        if current_time > self._last_update_time:
            time_increment = current_time - self._last_update_time
            self.waiting_time += time_increment
            self._last_update_time = current_time
    
    def depart(self, current_time: float):
        """Mark vehicle as departed"""
        self.departure_time = current_time
        self.travel_time = current_time - self.arrival_time
    
    @property
    def is_waiting(self) -> bool:
        """Check if vehicle is still waiting"""
        return self.departure_time is None
    
    @property
    def waiting_duration(self) -> float:
        """Get current waiting duration"""
        return self.waiting_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vehicle to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.vehicle_type.value,
            'arrival_time': self.arrival_time,
            'road': self.road,
            'waiting_time': self.waiting_time,
            'travel_time': self.travel_time,
            'departure_time': self.departure_time,
            'priority': self.priority,
            'attributes': self.attributes
        }
    
    def __repr__(self) -> str:
        return f"Vehicle(id={self.id}, type={self.vehicle_type.value}, road={self.road}, wait={self.waiting_time:.1f}s)"