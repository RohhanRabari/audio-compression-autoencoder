import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_encoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)                                  # 128 x 726 x 1     Input
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)    # 128 x 726 x 32    Conv
    x = layers.MaxPooling2D(pool_size=(2,2))(x)                                 # 64 x 363 x 32     MaxPool
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)         # 64 x 363 x 64     Conv
    x = layers.MaxPooling2D(pool_size=(2,2))(x)                                 # 32 x 181 x 64     MaxPool
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)        # 32 x 181 x 128    Conv
    

    
    
    # x = layers.Flatten()(x)                                                     # 73856             Flatten
    # # shape_before_flattening = tf.keras.backend.int_shape(x)[1:] 
    # latent = layers.Dense(latent_dim, activation='relu')(x)                     # 128               Dense
    
    encoder = tf.keras.Model(inputs, x, name="encoder")
    return encoder

if __name__ == '__main__':
    
    encoder = build_encoder((128, 726, 1), 128)
    
    encoder.summary()
