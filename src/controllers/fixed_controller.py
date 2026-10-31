"""
Fixed Controller - Always uses fixed green times
"""

from .base_controller import BaseController


class FixedController(BaseController):
    """
    Fixed-time traffic controller
    
    Simply alternates phases with fixed green times
    """
    
    def __init__(self, intersection, green_time: float = 30.0):
        """
        Initialize fixed controller
        
        Args:
            intersection: Intersection to control
            green_time: Fixed green time in seconds
        """
        super().__init__(intersection)
        self.green_time = green_time
        self.name = "Fixed Controller"
        self._phase_elapsed = 0.0
    
    def decide(self, current_time: float) -> bool:
        """
        Decide to switch after fixed green time
        
        Args:
            current_time: Current simulation time
            
        Returns:
            bool: True if phase should be switched
        """
        self._phase_elapsed += 1.0  # Assuming 1 second steps
        
        if self._phase_elapsed >= self.green_time:
            self._phase_elapsed = 0.0
            return True
        
        return False
    
    def get_phase_duration(self) -> float:
        """Get current phase duration"""
        return self.green_time
    
    def reset(self):
        """Reset the controller state"""
        self._phase_elapsed = 0.0