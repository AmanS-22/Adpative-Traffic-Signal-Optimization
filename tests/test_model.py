"""
Unit tests for Deep Learning Model
Run with: python -m pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import tempfile
from src.model.network import build_mlp, compile_model, get_callbacks
from src.model.train import ModelTrainer


class TestModel:
    """Test model classes"""
    
    def test_build_mlp(self):
        """Test MLP building"""
        model = build_mlp(input_dim=13, num_classes=3)
        
        assert model is not None
        # Check that model has layers (at least input, hidden, output)
        assert len(model.layers) >= 3
        
        # Test forward pass
        x = np.random.randn(1, 13)
        y = model.predict(x, verbose=0)
        assert y.shape == (1, 3)
        assert np.sum(y) == pytest.approx(1.0, abs=0.01)
    
    def test_compile_model(self):
        """Test model compilation"""
        model = build_mlp()
        compiled = compile_model(model)
        
        assert compiled is not None
        assert compiled.optimizer is not None
        assert compiled.loss == 'sparse_categorical_crossentropy'
    
    def test_get_callbacks(self):
        """Test callback creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = get_callbacks(
                model_path=os.path.join(tmpdir, 'model.keras'),
                patience=10
            )
            
            assert len(callbacks) == 4
            assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)
            assert isinstance(callbacks[1], tf.keras.callbacks.ReduceLROnPlateau)
            assert isinstance(callbacks[2], tf.keras.callbacks.ModelCheckpoint)
            assert isinstance(callbacks[3], tf.keras.callbacks.TensorBoard)
    
    def test_trainer_creation(self):
        """Test ModelTrainer creation"""
        trainer = ModelTrainer()
        
        assert trainer is not None
        assert trainer.input_dim == 13
        assert trainer.num_classes == 3
        assert trainer.model is None
        assert trainer.scaler is None
    
    def test_prepare_data(self):
        """Test data preparation"""
        trainer = ModelTrainer()
        
        # Create synthetic data
        n_samples = 100
        X = np.random.randn(n_samples, 13)
        y = np.random.randint(0, 3, n_samples)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(13)])
        df['action'] = y
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
        
        # Check splits
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_val) + len(X_test) == n_samples
    
    def test_normalize_data(self):
        """Test data normalization"""
        trainer = ModelTrainer()
        
        # Create synthetic data
        X_train = np.random.randn(100, 13)
        X_val = np.random.randn(50, 13)
        X_test = np.random.randn(50, 13)
        
        X_train_norm, X_val_norm, X_test_norm = trainer.normalize_data(X_train, X_val, X_test)
        
        assert X_train_norm.shape == X_train.shape
        assert X_val_norm.shape == X_val.shape
        assert X_test_norm.shape == X_test.shape
        assert trainer.scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])