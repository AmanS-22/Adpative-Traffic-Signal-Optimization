"""
Neural Network Architecture for Traffic Signal Control
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, Optional, List, Tuple


def build_mlp(
    input_dim: int = 13,
    num_classes: int = 3,
    hidden_layers: List[int] = None
) -> models.Sequential:
    """
    Build MLP for traffic signal control
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes (actions)
        hidden_layers: List of hidden layer sizes
        
    Returns:
        Compiled Keras model
    """
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers with regularization
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2 + i * 0.05))  # Increasing dropout
    
    # Output layer (3 actions: keep, switch, extend)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(
    model: models.Sequential,
    learning_rate: float = 0.001
) -> models.Sequential:
    """
    Compile the model with appropriate optimizer and loss
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
        
    Returns:
        Compiled model
    """
    # Learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(
    model_path: str = "model/best_model.keras",
    patience: int = 15
) -> List[tf.keras.callbacks.Callback]:
    """
    Get training callbacks
    
    Args:
        model_path: Path to save the best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks_list