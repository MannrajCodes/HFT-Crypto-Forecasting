import tensorflow as tf
import keras
from keras import layers, models
from config.config import settings

class ForecastModel:
    def __init__(self):
        self.sequence_length = settings.sequence_length
        self.features = settings.features
        self.horizon_step = settings.forecast_horizon
        self.learning_rate = settings.learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.sequence_length, self.features))
        
        # 1. Spatial Features (Standard Fast CNN)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # 2. Temporal Features (Standard Fast GRU)
        x = layers.GRU(64, return_sequences=False)(x)
        
        # 3. Dense Neural network
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.horizon_step, activation='linear')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # UPGRADE: Huber Loss tracks curves much smoother than MSE
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
        
        return model

    def predict(self, x_input):
        return self.model.predict(x_input, verbose=0)

    def get_model(self):
        return self.model

forecast_model = ForecastModel()