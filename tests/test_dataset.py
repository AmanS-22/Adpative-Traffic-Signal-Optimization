"""
Unit tests for Dataset Generator
Run with: python -m pytest tests/test_dataset.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.simulation.intersection import Intersection
from src.dataset.generate_dataset import DatasetGenerator


class TestDatasetGenerator:
    """Test DatasetGenerator class"""
    
    def test_dataset_generator_creation(self):
        """Test creating dataset generator"""
        generator = DatasetGenerator(seed=42)
        
        assert generator.seed == 42
        assert generator.samples == []
        assert len(generator.feature_names) == 14  # 13 features + action
    
    def test_get_state_features(self):
        """Test feature extraction"""
        generator = DatasetGenerator()
        intersection = Intersection()
        
        features = generator.get_state_features(intersection, 0.0)
        
        assert len(features) == 13
        assert features[0] >= 0  # n_queue
        assert features[1] >= 0  # s_queue
        assert features[2] >= 0  # e_queue
        assert features[3] >= 0  # w_queue
        assert features[8] in [0.0, 1.0]  # phase
    
    def test_evaluate_action(self):
        """Test action evaluation"""
        generator = DatasetGenerator()
        intersection = Intersection()
        
        costs = []
        for action in [0, 1, 2]:
            cost = generator.evaluate_action(intersection, action, 0.0)
            costs.append(cost)
        
        assert len(costs) == 3
        assert all(c >= 0 for c in costs)
    
    def test_get_optimal_action(self):
        """Test optimal action selection"""
        generator = DatasetGenerator()
        intersection = Intersection()
        
        action = generator.get_optimal_action(intersection, 0.0)
        
        assert action in [0, 1, 2]
    
    def test_generate_samples_small(self):
        """Test generating a small number of samples"""
        generator = DatasetGenerator(seed=42)
        
        df = generator.generate_samples(
            num_samples=100,  # Small for testing
            duration_per_scenario=50.0,
            sampling_interval=5.0
        )
        
        # Should have at least 100 samples (allow some tolerance)
        assert len(df) >= 90, f"Expected at least 90 samples, got {len(df)}"
        assert 'action' in df.columns
        assert all(col in df.columns for col in generator.feature_names)
    
    def test_save_load_dataset(self):
        """Test saving and loading dataset"""
        generator = DatasetGenerator(seed=42)
        
        # Generate small dataset
        df = generator.generate_samples(
            num_samples=50,
            duration_per_scenario=30.0,
            sampling_interval=5.0
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            generator.save_dataset(df, filepath)
            
            # Load and verify
            loaded_df = generator.load_dataset(filepath)
            
            assert len(loaded_df) == len(df)
            assert all(col in loaded_df.columns for col in generator.feature_names)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_action_distribution(self):
        """Test that dataset has varied actions"""
        generator = DatasetGenerator(seed=42)
        
        df = generator.generate_samples(
            num_samples=200,
            duration_per_scenario=50.0,
            sampling_interval=5.0
        )
        
        # Should have all actions
        actions = df['action'].unique()
        assert len(actions) >= 2  # At least 2 different actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])