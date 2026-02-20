import numpy as np
import logging
import os
from collections import deque
from sklearn.metrics import r2_score
from models.improved_models import forecast_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineTrainer:
    def __init__(self):
        self.model = forecast_model.get_model()
        
        # UPGRADE: Experience Replay Buffer
        # We hold the last 128 seconds of data so the model learns a smooth curve, not a sudden step.
        self.replay_x = deque(maxlen=128)
        self.replay_y = deque(maxlen=128)
        
        self.train_frequency = 4  # Gently update the weights every 4 ticks
        self.step_count = 0
        
        self.weights_path = "storage/model_checkpoints/latest_model.weights.h5"
        self.history_true = deque(maxlen=300)
        self.history_pred = deque(maxlen=300)
        
        self._load_existing_weights()

    def _load_existing_weights(self):
        try:
            self.model.load_weights(self.weights_path)
            logger.info("Loaded existing model weights.")
        except Exception:
            logger.info("No existing weights found. Starting fresh.")

    def add_sample(self, x_seq, y_target):
        # We append to the rolling buffer (it automatically pushes old data out)
        self.replay_x.append(x_seq)
        self.replay_y.append(y_target)
        self.step_count += 1
        
        # Once we have 64 steps of history, we train every 4 seconds
        if len(self.replay_x) >= 64 and self.step_count % self.train_frequency == 0:
            self.train_incremental()

    def train_incremental(self):
        x_train = np.array(self.replay_x)
        y_train = np.array(self.replay_y)
        
        # Train for just ONE epoch. This "glides" the weights down smoothly during a crash
        # instead of causing a 90-degree violent staircase drop.
        history = self.model.fit(
            x_train, y_train, 
            epochs=1, 
            batch_size=32, 
            verbose=0
        )
        
        # Predict the latest batch for our R2 scoring
        y_pred = self.model.predict(x_train[-self.train_frequency:], verbose=0)
        
        for true_val, pred_val in zip(y_train[-self.train_frequency:], y_pred):
            self.history_true.append(true_val[0])
            self.history_pred.append(pred_val[0])
            
        if len(self.history_true) > 64:
            try:
                current_r2 = r2_score(self.history_true, self.history_pred)
            except:
                current_r2 = 0.0
        else:
            current_r2 = 0.0 
            
        loss = history.history.get('loss', [0])[0]
        mae = history.history.get('mae', [0])[0]
        
        logger.info(f"Smooth Training. Loss (Huber): {loss:.4f}, MAE: {mae:.4f}, Rolling R²: {current_r2:.4f}")
        
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        self.model.save_weights(self.weights_path)

online_trainer = OnlineTrainer()