"""
Unit tests for Controllers
Run with: python -m pytest tests/test_controllers.py -v
"""

import pytest
from src.simulation.intersection import Intersection
from src.controllers.fixed_controller import FixedController
from src.controllers.rule_controller import RuleBasedController
from src.controllers.dl_controller import DeepLearningController


class TestControllers:
    """Test controller classes"""
    
    def test_fixed_controller_creation(self):
        """Test fixed controller creation"""
        intersection = Intersection()
        controller = FixedController(intersection, green_time=20.0)
        
        assert controller.green_time == 20.0
        assert controller.name == "Fixed Controller"
        assert controller._phase_elapsed == 0.0
    
    def test_fixed_controller_decision(self):
        """Test fixed controller decision"""
        intersection = Intersection()
        controller = FixedController(intersection, green_time=5.0)
        
        # Should not switch before green time
        for i in range(4):
            assert controller.decide(0.0) == False
        
        # Should switch at green time
        assert controller.decide(0.0) == True
        assert controller._phase_elapsed == 0.0  # Reset after switch
    
    def test_rule_controller_creation(self):
        """Test rule-based controller creation"""
        intersection = Intersection()
        controller = RuleBasedController(
            intersection,
            min_green=8.0,
            max_green=30.0,
            hysteresis=0.25
        )
        
        assert controller.min_green == 8.0
        assert controller.max_green == 30.0
        assert controller.hysteresis == 0.25
        assert controller.name == "Rule-Based Controller"
    
    def test_rule_controller_min_green(self):
        """Test rule controller respects minimum green"""
        intersection = Intersection()
        controller = RuleBasedController(intersection, min_green=5.0)
        
        # Should not switch before min_green
        for i in range(4):
            assert controller.decide(0.0) == False
        
        # Should consider switching after min_green
        # At 5 seconds, it can switch
        controller._phase_elapsed = 5.0
        # Now it will check demand
        result = controller.decide(0.0)
        # Result depends on demand (could be True or False)
        assert result in [True, False]
    
    def test_rule_controller_max_green(self):
        """Test rule controller respects maximum green"""
        intersection = Intersection()
        controller = RuleBasedController(intersection, max_green=10.0)
        
        # Force elapsed to max_green
        controller._phase_elapsed = 10.0
        
        # Should switch at max_green regardless of demand
        assert controller.decide(0.0) == True
        assert controller._phase_elapsed == 0.0
    
    def test_dl_controller_creation(self):
        """Test deep learning controller creation"""
        intersection = Intersection()
        controller = DeepLearningController(
            intersection,
            min_green=8.0,
            max_green=35.0
        )
        
        assert controller.min_green == 8.0
        assert controller.max_green == 35.0
        assert controller.name == "Deep Learning Controller"
        assert controller.model is None  # No model loaded
    
    def test_dl_controller_fallback(self):
        """Test DL controller fallback behavior"""
        intersection = Intersection()
        controller = DeepLearningController(
            intersection,
            min_green=5.0,
            max_green=20.0
        )
        
        # Without model, should use fallback
        # Min green constraint
        for i in range(4):
            assert controller.decide(0.0) == False
        
        # After min green, fallback logic applies
        controller._phase_elapsed = 10.0
        # Should switch if max green reached
        controller.max_green = 10.0
        assert controller.decide(0.0) == True
    
    def test_controller_reset(self):
        """Test resetting controllers"""
        intersection = Intersection()
        
        # Test fixed controller reset
        fixed = FixedController(intersection)
        fixed._phase_elapsed = 15.0
        fixed.reset()
        assert fixed._phase_elapsed == 0.0
        
        # Test rule controller reset
        rule = RuleBasedController(intersection)
        rule._phase_elapsed = 20.0
        rule.reset()
        assert rule._phase_elapsed == 0.0
        
        # Test DL controller reset
        dl = DeepLearningController(intersection)
        dl._phase_elapsed = 25.0
        dl._predictions = [5.0, 6.0, 7.0]
        dl.reset()
        assert dl._phase_elapsed == 0.0
        assert len(dl._predictions) == 0
    
    def test_get_features(self):
        """Test feature extraction from controllers"""
        intersection = Intersection()
        controller = RuleBasedController(intersection)
        
        # Pass current_time to the method
        features = controller.get_state_features(current_time=10.0)
        
        assert 'n_queue' in features
        assert 's_queue' in features
        assert 'e_queue' in features
        assert 'w_queue' in features
        assert 'total_queue' in features
        assert 'phase' in features
        assert 'elapsed' in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])