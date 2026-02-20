import numpy as np
import collections
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from config.config import settings

class StreamingPipeline:
    def __init__(self):
        self.window_size = settings.sequence_length
        self.feature_count = settings.features
        self.data_buffer = collections.deque(maxlen=self.window_size)
        
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False
        
        # Welford's Algorithm state for Anomaly Detection
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def _update_welford(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def _get_variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / self.n

    def detect_anomaly(self, value: float, threshold: float = 3.0) -> bool:
        if self.n < 10:
            self._update_welford(value)
            return False
            
        variance = self._get_variance()
        std_dev = np.sqrt(variance)
        
        if std_dev > 0 and abs(value - self.mean) > (threshold * std_dev):
            self._update_welford(value)
            return True
            
        self._update_welford(value)
        return False

    def ingest_new_data(self, data_point: Dict[str, Any]) -> Tuple[bool, Optional[np.ndarray]]:
        target_value = float(data_point['views'])
        feature_value = float(data_point['unique_visitors'])
        
        is_anomaly = self.detect_anomaly(target_value)
        self.data_buffer.append([target_value, feature_value])
        
        if len(self.data_buffer) == self.window_size:
            sequence = np.array(self.data_buffer)
            
            # --- RESEARCH PAPER: SWT NOISE FILTERING LAYER ---
            import pywt
            
            # 1. Extract just the target variable (The Carbon/Views line)
            target_signal = sequence[:, 0]
            
            # 2. Perform 2-Level Stationary Wavelet Transform (Haar Wavelet)
            coeffs = pywt.swt(target_signal, 'haar', level=2)
            
            # 3. Filter out the noise (Zero out the detail coefficients 'cD')
            clean_coeffs = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
            
            # 4. Reconstruct the clean signal
            clean_target = pywt.iswt(clean_coeffs, 'haar')
            
            # 5. Overwrite the noisy data with the clean wave before the neural network sees it!
            sequence[:, 0] = clean_target
            # --------------------------------------------------
            
            self.scaler.partial_fit(sequence)
            self.is_scaler_fitted = True
            
            scaled_seq = self.scaler.transform(sequence)
            return is_anomaly, np.array([scaled_seq])
            
        return is_anomaly, None # <--- THIS is what was missing!
pipeline = StreamingPipeline()