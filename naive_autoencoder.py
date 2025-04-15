import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ConvLSTM2D, Reshape, TimeDistributed
from tensorflow.keras.models import Model

def build_conv_lstm_autoencoder(sequence_length=3, height=64, width=64, channels=3):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    input_shape = (sequence_length, height, width, channels)
    inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = TimeDistributed(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))(x)
    outputs = TimeDistributed(Conv2DTranspose(channels, (3, 3), activation='sigmoid', padding='same'))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
