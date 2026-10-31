"""
Base Controller - Abstract base class for all traffic controllers
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from src.simulation.intersection import Intersection, SignalPhase


class BaseController(ABC):
    """
    Abstract base class for traffic controllers
    """
    
    def __init__(self, intersection: Intersection):
        """
        Initialize controller
        
        Args:
            intersection: Intersection to control
        """
        self.intersection = intersection
        self.name = "Base Controller"
    
    @abstractmethod
    def decide(self, current_time: float) -> bool:
        """
        Make a decision about whether to switch phase
        
        Args:
            current_time: Current simulation time
            
        Returns:
            bool: True if phase should be switched
        """
        pass
    
    @abstractmethod
    def get_phase_duration(self) -> float:
        """
        Get the duration of the current phase
        
        Returns:
            float: Phase duration in seconds
        """
        pass
    
    def get_state_features(self, current_time: float) -> Dict[str, float]:
        """
        Get features for AI/decision making
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dict of feature names to values
        """
        queues = self.intersection.get_queue_lengths()
        return {
            'n_queue': queues.get('N', 0),
            's_queue': queues.get('S', 0),
            'e_queue': queues.get('E', 0),
            'w_queue': queues.get('W', 0),
            'total_queue': sum(queues.values()),
            'phase': 1 if self.intersection.phase == SignalPhase.NS_GO else 0,
            'elapsed': self.intersection.get_elapsed_green(current_time)
        }
    
    def reset(self):
        """
        Reset the controller state
        Override in child classes if needed
        """
        pass