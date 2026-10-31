"""
Training pipeline for the Deep Learning model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import callbacks
import pickle
from typing import Dict, Tuple, Optional

from .network import build_mlp, compile_model, get_callbacks


class ModelTrainer:
    """
    Handles training and evaluation of the Deep Learning model
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        num_classes: int = 3,
        model_path: str = "model/best_model.keras",
        scaler_path: str = "model/scaler.pkl"
    ):
        """
        Initialize model trainer
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Create model directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
    
    def load_dataset(self, filepath: str = "dataset/traffic_dataset.csv") -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with dataset
        """
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} features")
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with dataset
            test_size: Test set size
            val_size: Validation set size
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'action']
        X = df[feature_cols].values
        y = df['action'].values
        
        print(f"\n📊 Data Shape:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        print(f"  Action distribution:")
        print(pd.Series(y).value_counts())
        
        # Split into train + val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split temp into train and val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Split Sizes:")
        print(f"  Training: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.1f}%)")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
        """
        Normalize features using StandardScaler
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Normalized arrays
        """
        self.scaler = StandardScaler()
        
        # Fit on training data only
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        print("\n✅ Data normalized using StandardScaler")
        print(f"  Mean: {self.scaler.mean_[:5]}...")
        print(f"  Std: {self.scaler.scale_[:5]}...")
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the model
        
        Returns:
            Compiled Keras model
        """
        self.model = build_mlp(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_layers=[128, 64, 32]
        )
        
        self.model = compile_model(self.model, learning_rate=0.001)
        
        print("\n🧠 Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64
    ) -> tf.keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Get callbacks
        callbacks_list = get_callbacks(self.model_path)
        
        print(f"\n🚀 Training model for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print(f"\n✅ Training complete!")
        print(f"  Final Training Accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            self.load_model()
        
        print("\n📊 Evaluating model on test set...")
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['Keep', 'Switch', 'Extend'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\n📊 Test Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\n  Classification Report:")
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"    {class_name}:")
                print(f"      Precision: {metrics['precision']:.4f}")
                print(f"      Recall: {metrics['recall']:.4f}")
                print(f"      F1-score: {metrics['f1-score']:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: str = "results/training_history.png"):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("⚠️ No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "results/confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Keep', 'Switch', 'Extend'],
            yticklabels=['Keep', 'Switch', 'Extend'],
            ax=ax
        )
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
        plt.show()
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.model is None:
            print("⚠️ No model to save")
            return
        
        # Save model
        self.model.save(self.model_path)
        print(f"✅ Model saved to {self.model_path}")
        
        # Save scaler
        if self.scaler is not None:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler saved to {self.scaler_path}")
    
    def load_model(self):
        """Load the trained model and scaler"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model not found at {self.model_path}")
        
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {self.scaler_path}")
        else:
            print(f"⚠️ Scaler not found at {self.scaler_path}")
    
    def run_pipeline(
        self,
        dataset_path: str = "dataset/traffic_dataset.csv",
        epochs: int = 100,
        batch_size: int = 64
    ) -> Dict:
        """
        Run the complete training pipeline
        
        Args:
            dataset_path: Path to dataset
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Evaluation results
        """
        print("=" * 60)
        print("🧠 DEEP LEARNING MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # 1. Load dataset
        df = self.load_dataset(dataset_path)
        
        # 2. Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        
        # 3. Normalize data
        X_train_norm, X_val_norm, X_test_norm = self.normalize_data(X_train, X_val, X_test)
        
        # 4. Build model
        self.build_model()
        
        # 5. Train model
        history = self.train(X_train_norm, y_train, X_val_norm, y_val, epochs, batch_size)
        
        # 6. Evaluate model
        results = self.evaluate(X_test_norm, y_test)
        
        # 7. Plot results
        self.plot_training_history()
        self.plot_confusion_matrix(results['confusion_matrix'])
        
        # 8. Save model
        self.save_model()
        
        return results


if __name__ == "__main__":
    # Run training pipeline
    trainer = ModelTrainer()
    results = trainer.run_pipeline(
        dataset_path="dataset/traffic_dataset.csv",
        epochs=50,
        batch_size=64
    )