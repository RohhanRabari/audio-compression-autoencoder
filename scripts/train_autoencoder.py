import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam #legacy runs faster on M1/M2 macs
from keras.losses import MeanSquaredError
import yaml
from models.autoencoder import build_autoencoder
from utils.data_loader import load_spectrograms
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
spectrograms = load_spectrograms('/Users/ronin/AI/audio-compression-autoencoder/data/microsoft-speech-corpus/processed/gu-in-Test')
spectrograms = np.asarray(spectrograms).astype(np.float32)

print(np.shape(spectrograms))

# import sys
# sys.exit()

# dataset = tf.data.Dataset.from_tensor_slices(spectrograms).batch(config['training']['batch_size'])

X_train, X_val, y_train, y_val = train_test_split(spectrograms, spectrograms, test_size=0.2, random_state=42)

autoencoder = build_autoencoder(input_shape=config['model']['input_shape'], latent_dim=config['model']['latent_dim'])
autoencoder.compile(optimizer=Adam(learning_rate=config['training']['learning_rate']),
                    loss=MeanSquaredError(),
                    metrics=[keras.metrics.Accuracy(),
                             keras.metrics.Precision(),
                             keras.metrics.Recall()])

autoencoder.summary()

checkpoint_callback = ModelCheckpoint(filepath=os.path.join(config['training']['model_save_path']),
                                      monitor='val_loss',
                                      save_best_only=True,
                                      mode='auto',
                                      save_weights_only=False)

tensorboard_callback = TensorBoard(log_dir=config['training']['tensorboard_log_dir'], 
                                   update_freq='epoch',
                                   write_graph=True)

# Training
autoencoder.fit(X_train, y_train,
                epochs=config['training']['epochs'],
                validation_data=(X_val, y_val),
                callbacks=[checkpoint_callback, tensorboard_callback])

# Save model

final_model_path = config['training']['final_model_save_path']
if not os.path.exists(os.path.dirname(final_model_path)):
    os.makedirs(os.path.dirname(final_model_path))
autoencoder.save(final_model_path)

