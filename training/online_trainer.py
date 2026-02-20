import numpy as np
import logging
import os
from collections import deque
from sklearn.metrics import r2_score
from models.improved_models import forecast_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineTrainer:
    def __init__(self, batch_size=32):
        self.model = forecast_model.get_model()
        self.batch_size = batch_size
        self.x_buffer = []
        self.y_buffer = []
        self.weights_path = "storage/model_checkpoints/latest_model.weights.h5"
        
        # THE FIX: A rolling memory of the last 300 predictions for accurate R2 math
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
        self.x_buffer.append(x_seq)
        self.y_buffer.append(y_target)
        
        if len(self.x_buffer) >= self.batch_size:
            self.train_incremental()

    def train_incremental(self):
        if not self.x_buffer:
            return
            
        logger.info(f"Starting incremental training on {len(self.x_buffer)} new samples...")
        
        x_train = np.array(self.x_buffer)
        y_train = np.array(self.y_buffer)
        
        # 1. Train the model on the new batch (Stable 10 epochs)
        history = self.model.fit(
            x_train, 
            y_train, 
            epochs=10, 
            batch_size=self.batch_size, 
            verbose=0
        )
        
        loss = history.history['loss'][0]
        mae = history.history['mae'][0]
        
        # 2. Predict on the batch to log history
        y_pred = self.model.predict(x_train, verbose=0)
        
        # Append to our rolling 5-minute memory
        for true_val, pred_val in zip(y_train, y_pred):
            self.history_true.append(true_val[0])
            self.history_pred.append(pred_val[0])
            
        # 3. CALCULATE R2 SCORE ON THE LARGER WINDOW
        # Wait until we have at least 64 data points to avoid the divide-by-zero bug
        if len(self.history_true) > 64:
            try:
                current_r2 = r2_score(self.history_true, self.history_pred)
            except:
                current_r2 = 0.0
        else:
            current_r2 = 0.0 # Warming up
            
        logger.info(f"Training complete. Loss (MSE): {loss:.4f}, MAE: {mae:.4f}, Rolling R² Score: {current_r2:.4f}")
        
        # Save weights
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        self.model.save_weights(self.weights_path)
        
        # Clear batch buffer
        self.x_buffer = []
        self.y_buffer = []

online_trainer = OnlineTrainer()