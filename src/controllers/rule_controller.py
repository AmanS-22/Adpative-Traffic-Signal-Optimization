"""
Rule-Based Controller - Uses heuristics to decide when to switch
"""

from .base_controller import BaseController
from src.simulation.intersection import SignalPhase


class RuleBasedController(BaseController):
    """
    Rule-based adaptive traffic controller
    
    Switches based on:
    - Minimum green time (safety)
    - Maximum green time (fairness)
    - Queue length comparison (efficiency)
    """
    
    def __init__(
        self,
        intersection,
        min_green: float = 10.0,
        max_green: float = 40.0,
        hysteresis: float = 0.2
    ):
        """
        Initialize rule-based controller
        
        Args:
            intersection: Intersection to control
            min_green: Minimum green time in seconds
            max_green: Maximum green time in seconds
            hysteresis: Hysteresis threshold for switching
        """
        super().__init__(intersection)
        self.min_green = min_green
        self.max_green = max_green
        self.hysteresis = hysteresis
        self.name = "Rule-Based Controller"
        self._phase_elapsed = 0.0
    
    def decide(self, current_time: float) -> bool:
        """
        Decide based on queue lengths and time constraints
        
        Args:
            current_time: Current simulation time
            
        Returns:
            bool: True if phase should be switched
        """
        # Update elapsed time
        self._phase_elapsed += 1.0
        
        # Minimum green constraint
        if self._phase_elapsed < self.min_green:
            return False
        
        # Get demands for both phases
        current_phase = self.intersection.phase
        other_phase = SignalPhase.EW_GO if current_phase == SignalPhase.NS_GO else SignalPhase.NS_GO
        
        current_demand = self.intersection.get_phase_demand(current_phase)
        other_demand = self.intersection.get_phase_demand(other_phase)
        
        # Check if should switch
        should_switch = False
        
        # 1. Maximum green reached - must switch
        if self._phase_elapsed >= self.max_green:
            should_switch = True
        
        # 2. Hysteresis switch - other phase has significantly higher demand
        elif other_demand > current_demand * (1 + self.hysteresis):
            should_switch = True
        
        # 3. Starvation prevention - other phase has vehicles and current has none
        elif other_demand > 0 and current_demand == 0:
            should_switch = True
        
        if should_switch:
            self._phase_elapsed = 0.0
            return True
        
        return False
    
    def get_phase_duration(self) -> float:
        """Get current phase duration"""
        return self._phase_elapsed
    
    def reset(self):
        """Reset the controller state"""
        self._phase_elapsed = 0.0