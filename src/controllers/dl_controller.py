"""
Deep Learning Controller - Uses the EXISTING trained model.

Model contract:
  Input:  13 features (see FEATURE_NAMES)
  Output: 3-class softmax  →  [KEEP, SWITCH, EXTEND]
  Scaler: StandardScaler at model/scaler.pkl
"""

import os
import pickle
import numpy as np
from typing import Optional, Dict, List

from .base_controller import BaseController
from src.simulation.intersection import SignalPhase


# Toggle this to enable/disable DL debug logging
DEBUG_DL = True


class DeepLearningController(BaseController):

    FEATURE_NAMES = [
        'n_queue', 's_queue', 'e_queue', 'w_queue',
        'n_arrival_rate', 's_arrival_rate', 'e_arrival_rate', 'w_arrival_rate',
        'phase', 'elapsed_green',
        'avg_wait_time', 'max_wait_time', 'total_queue',
    ]
    ACTION_NAMES = ['KEEP', 'SWITCH', 'EXTEND']

    def __init__(
        self,
        intersection,
        model_path: str = "model/best_model.keras",
        scaler_path: str = "model/scaler.pkl",
        min_green: float = 10.0,
        max_green: float = 40.0,
        confidence_threshold: float = 0.40,
        debug: bool = DEBUG_DL,
    ):
        super().__init__(intersection)
        self.name = "Deep Learning Controller"

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.min_green = min_green
        self.max_green = max_green
        self.confidence_threshold = confidence_threshold
        self.debug = debug

        self._phase_elapsed = 0.0
        self._predictions: List[Dict] = []
        self._last_prediction: Optional[Dict] = None
        self._decision_counter = 0

        self.model = None
        self.scaler = None

        self.stats = {
            'decisions': 0,
            'keep': 0,
            'switch': 0,
            'extend': 0,
            'low_conf': 0,
            'model_errors': 0,
            'max_green_force': 0,
        }

        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import tensorflow as tf
        except ImportError:
            print("[DL] ❌ TensorFlow not installed")
            return

        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"[DL] ✅ Model loaded: {self.model_path}")
                print(f"[DL]    Input:  {self.model.input_shape}")
                print(f"[DL]    Output: {self.model.output_shape}")
            except Exception as e:
                print(f"[DL] ❌ Model load failed: {e}")
                self.model = None
        else:
            print(f"[DL] ❌ Model not found: {self.model_path}")

        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"[DL] ✅ Scaler loaded: {self.scaler_path}")
                if hasattr(self.scaler, 'mean_'):
                    print(f"[DL]    mean[9]={self.scaler.mean_[9]:.4f}  "
                          f"scale[9]={self.scaler.scale_[9]:.4f}")
            except Exception as e:
                print(f"[DL] ⚠️ Scaler load failed: {e}")
                self.scaler = None
        else:
            print(f"[DL] ⚠️ Scaler not found: {self.scaler_path}")

    # ------------------------------------------------------------------
    def _extract_features(self) -> np.ndarray:
        q = self.intersection.get_queue_lengths()
        a = self.intersection.get_arrival_rates()
        phase_val = 1.0 if self.intersection.phase == SignalPhase.NS_GO else 0.0

        return np.array([[
            q.get('N', 0) / 50.0,
            q.get('S', 0) / 50.0,
            q.get('E', 0) / 50.0,
            q.get('W', 0) / 50.0,
            a.get('N', 0) / 10.0,
            a.get('S', 0) / 10.0,
            a.get('E', 0) / 10.0,
            a.get('W', 0) / 10.0,
            phase_val,
            self._phase_elapsed / 60.0,
            self.intersection.get_average_waiting_time() / 60.0,
            self.intersection.get_max_waiting_time() / 60.0,
            sum(q.values()) / 100.0,
        ]], dtype=np.float32)

    # ------------------------------------------------------------------
    def _predict(self) -> Optional[Dict]:
        if self.model is None:
            return None

        raw = self._extract_features()
        if self.scaler is not None:
            try:
                scaled = self.scaler.transform(raw)
            except Exception as e:
                if self.debug:
                    print(f"[DL] ⚠️ Scaler error: {e}")
                scaled = raw
        else:
            scaled = raw

        try:
            probs = self.model.predict(scaled, verbose=0)[0]
        except Exception as e:
            if self.debug:
                print(f"[DL] ❌ Prediction error: {e}")
            self.stats['model_errors'] += 1
            return None

        action = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return {
            'action': action,
            'confidence': confidence,
            'probs': probs.tolist(),
            'raw': raw[0].tolist(),
            'scaled': scaled[0].tolist() if hasattr(scaled, 'tolist') else list(scaled[0]),
        }

    # ------------------------------------------------------------------
    def decide(self, current_time: float) -> bool:
        self._phase_elapsed += 1.0
        self.stats['decisions'] += 1
        self._decision_counter += 1

        # Safety: minimum green
        if self._phase_elapsed < self.min_green:
            return False

        # Safety: force switch at max green
        if self._phase_elapsed >= self.max_green:
            self.stats['max_green_force'] += 1
            self._reset_elapsed()
            if self.debug and self._decision_counter % 5 == 0:
                print(f"[DL] t={current_time:6.1f}s  MAX_GREEN forced switch")
            return True

        if self.model is None:
            return False

        pred = self._predict()
        if pred is None:
            return False

        self._last_prediction = pred
        self._predictions.append(pred)

        action = pred['action']
        conf = pred['confidence']
        probs = pred['probs']

        # Debug log every 5 decisions
        if self.debug and self._decision_counter % 5 == 0:
            q = self.intersection.get_queue_lengths()
            print(
                f"\n[DL] ─── Decision {self._decision_counter} ───"
                f"\n[DL]   state: N={q['N']} S={q['S']} E={q['E']} W={q['W']}"
                f"\n[DL]   input:  {[f'{x:.3f}' for x in pred['raw']]}"
                f"\n[DL]   scaled: {[f'{x:.3f}' for x in pred['scaled']]}"
                f"\n[DL]   predict: KEEP={probs[0]:.3f} SWITCH={probs[1]:.3f} EXTEND={probs[2]:.3f}"
                f"\n[DL]   action: {self.ACTION_NAMES[action]} (conf={conf:.3f})"
                f"\n[DL]   prev_phase: {self.intersection.phase.value}"
                f"\n[DL]   phase_changed: {action == 1}"
            )

        # Low confidence → hold
        if conf < self.confidence_threshold:
            self.stats['low_conf'] += 1
            return False

        if action == 0:  # KEEP
            self.stats['keep'] += 1
            return False

        if action == 1:  # SWITCH
            self.stats['switch'] += 1
            self._reset_elapsed()
            return True

        if action == 2:  # EXTEND
            self.stats['extend'] += 1
            self._phase_elapsed = max(self.min_green, self._phase_elapsed - 3.0)
            return False

        return False

    # ------------------------------------------------------------------
    def _reset_elapsed(self):
        self._phase_elapsed = 0.0

    def get_phase_duration(self) -> float:
        return self._phase_elapsed

    def get_predictions(self) -> List[Dict]:
        return self._predictions

    def get_diagnostics(self) -> Dict:
        return {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'model_path': self.model_path,
            'stats': self.stats.copy(),
            'last_prediction': self._last_prediction,
        }

    def reset(self):
        self._phase_elapsed = 0.0
        self._predictions = []
        self._last_prediction = None