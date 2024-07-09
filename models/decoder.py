import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_decoder(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(latent_inputs)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)
    
    decoder = tf.keras.Model(latent_dim, outputs, name="decoder")
    return decoder

if __name__ == '__main__':
    
    decoder = build_decoder(128)
    decoder.summary()