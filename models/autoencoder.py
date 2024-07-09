import tensorflow as tf
from tensorflow import keras
from keras import layers
from models.encoder import build_encoder
from models.decoder import build_decoder

def build_autoencoder(input_shape, latent_dim):
    # encoder = build_encoder(input_shape, latent_dim)
    # decoder = build_decoder(latent_dim)
    # autoencoder = tf.keras.Model(encoder.input, decoder(encoder.output), name="autoencoder")
    # return autoencoder
    
    inputs = tf.keras.Input(shape=input_shape)                                  # 128 x 726 x 1     Input
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)    # 128 x 726 x 32    Conv
    x = layers.MaxPooling2D(pool_size=(2,2),  padding='same')(x)                                 # 64 x 363 x 32     MaxPool
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)         # 64 x 363 x 64     Conv
    x = layers.MaxPooling2D(pool_size=(2,2),  padding='same')(x)                                 # 32 x 181 x 64     MaxPool
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)        # 32 x 181 x 128    Conv
    
    x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)
    
    autoencoder = tf.keras.Model(inputs, outputs, name="autoencoder")
    return autoencoder

# if __name__ == '__main__':
    # autoencoder = build_autoencoder((128, 728))
