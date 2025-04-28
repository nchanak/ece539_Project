# video_compression_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
import numpy as np

# ---------------------------
# Vector Quantizer Layer
# ---------------------------
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        initializer = tf.keras.initializers.VarianceScaling()
        self.embeddings = tf.Variable(
            initializer(shape=(self.embedding_dim, self.num_embeddings)),
            trainable=True
        )

    def call(self, inputs):
        flat = tf.reshape(inputs, [-1, self.embedding_dim])
        distances = (
            tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat, self.embeddings)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
        )
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, tf.transpose(self.embeddings))
        quantized = tf.reshape(quantized, tf.shape(inputs))

        loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2) + \
               self.commitment_cost * tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        self.add_loss(loss)

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized, tf.reshape(encoding_indices, tf.shape(inputs)[:-1])

def build_encoder(input_shape=(8, 64, 64, 3), latent_channels=32):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv3D(128, 5, strides=(1, 2, 2), padding='same')(inputs)  # → (8, 32, 32)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(128, 5, strides=(1, 2, 2), padding='same')(x)       # → (8, 16, 16)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = residual_block_3d(x, 128)

    # Temporal downsampling only (optional)
    x = layers.Conv3D(latent_channels, 3, strides=(2, 1, 1), padding='same')(x)  # → (4, 16, 16)
    return models.Model(inputs, x, name="Encoder")


# === Decoder ===
def build_decoder(latent_shape, output_channels=3):
    inputs = tf.keras.Input(shape=latent_shape)

    # Temporal upsampling (if done in encoder)
    x = layers.Conv3DTranspose(128, 3, strides=(2, 1, 1), padding='same')(inputs)  # → (8, 16, 16)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = residual_block_3d(x, 128)

    x = layers.Conv3DTranspose(128, 5, strides=(1, 2, 2), padding='same')(x)  # → (8, 32, 32)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3DTranspose(output_channels, 5, strides=(1, 2, 2), padding='same', activation='sigmoid')(x)  # → (8, 64, 64)
    return models.Model(inputs, x, name="Decoder")


# ---------------------------
# Full Autoencoder + Quantization
# ---------------------------
def build_vq_autoencoder(input_shape, num_embeddings=512, embedding_dim=64):
    encoder = build_encoder(input_shape, embedding_dim)
    decoder = build_decoder(encoder.output_shape[1:], 3)
    vq = VectorQuantizer(num_embeddings, embedding_dim)

    inputs = layers.Input(shape=input_shape)
    z = encoder(inputs)
    z_q = vq(z)
    recon = decoder(z_q)

    return models.Model(inputs, recon), encoder, vq


# === Residual Block (3D) ===
def residual_block_3d(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv3D(filters, kernel_size, padding='same', strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

import tensorflow as tf
from tensorflow.keras import layers, models

class PixelCNNPrior(tf.keras.Model):
    def __init__(self, input_shape, num_embeddings, num_filters=128, num_layers=8):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.pixelcnn = self._build_pixelcnn(input_shape, num_embeddings, num_filters, num_layers)

    def _build_pixelcnn(self, input_shape, num_embeddings, num_filters, num_layers):
        T, H, W = input_shape  # This now works because it's defined in class scope
        inputs = layers.Input(shape=(T, H, W, num_embeddings))

        x = layers.Conv3D(num_filters, kernel_size=1, padding='same', activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Conv3D(num_filters, kernel_size=3, padding='same', activation='relu')(x)

        logits = layers.Conv3D(num_embeddings, kernel_size=1, padding='same')(x)
        return models.Model(inputs, logits, name="PixelCNNPrior")


    def call(self, z_indices):
        z_onehot = tf.one_hot(z_indices, depth=self.num_embeddings)
        return self.pixelcnn(z_onehot)

    def log_prob(self, z_indices):
        """
        z_indices: shape (batch_size, T, H, W), integer indices into codebook
        Returns: log-probabilities of the given indices under the autoregressive prior
        """
        # One-hot encode before passing to pixelcnn
        z_onehot = tf.one_hot(z_indices, depth=self.num_embeddings, dtype=tf.float32)
        logits = self.pixelcnn(z_onehot)  # logits shape: (batch, T, H, W, num_embeddings)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        return tf.reduce_sum(log_probs * z_onehot, axis=-1)  # shape: (batch, T, H, W)
    
    def prob(self, z_indices):
        z_onehot = tf.one_hot(z_indices, depth=self.num_embeddings)
        logits = self.pixelcnn(z_onehot)
        probs = tf.nn.softmax(logits, axis=-1)
        return probs



class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, beta=1.0, pixelcnn_prior=None):
        super().__init__()
        self.beta = beta
        self.pixelcnn_prior = pixelcnn_prior  # must return logits or log-prob

    def call(self, y_true, y_pred):
        recon = y_pred["reconstruction"]
        z_q = y_pred["quantized"]
        z_indices = y_pred["z_indices"]

        # Distortion (MS-SSIM as negative similarity -> loss)

        distortion = 1 - tf.reduce_mean(ssim(
            tf.clip_by_value(y_true, 0, 1),
            tf.clip_by_value(y_pred["reconstruction"], 0, 1),
            max_val=1.0
        ))


        # Rate (log prob from PixelCNN prior)
        log_probs = self.pixelcnn_prior.log_prob(z_indices)  # custom method below
        rate = -tf.reduce_mean(log_probs)  # negative log likelihood (maximize prob)
        print("distortion:", distortion.numpy(), "rate:", rate.numpy())

        return distortion + self.beta * rate
    
def build_vqvae_with_prior(input_shape, encoder, decoder, vq_layer, pixelcnn_prior):
    inputs = tf.keras.Input(shape=input_shape)
    z_continuous = encoder(inputs)
    z_quantized, z_indices = vq_layer(z_continuous)
    recon = decoder(z_quantized)

    model = tf.keras.Model(inputs, {
        "reconstruction": recon,
        "quantized": z_quantized,
        "z_indices": z_indices
    })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=RateDistortionLoss(beta=1.0, pixelcnn_prior=pixelcnn_prior)
    )
    return model

class VQVAEModule(tf.keras.Model):
    def __init__(self, encoder, decoder, vq_layer, pixelcnn_prior, beta=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq_layer = vq_layer
        self.pixelcnn_prior = pixelcnn_prior
        self.beta = beta
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.distortion_tracker = tf.keras.metrics.Mean(name="distortion")
        self.rate_tracker = tf.keras.metrics.Mean(name="rate")

    def call(self, inputs):
        z_cont = self.encoder(inputs)
        z_q, z_idx = self.vq_layer(z_cont)
        recon = self.decoder(z_q)
        return {"reconstruction": recon, "quantized": z_q, "z_indices": z_idx}

    def train_step(self, data):
        x = data  # data is (batch,) in our tf.data.Dataset

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            recon = outputs["reconstruction"]
            z_indices = outputs["z_indices"]

            distortion = 1 - tf.reduce_mean(tf.image.ssim(
                tf.clip_by_value(x, 0, 1),
                tf.clip_by_value(recon, 0, 1),
                max_val=1.0
            ))


            log_probs = self.pixelcnn_prior.log_prob(z_indices)
            rate = -tf.reduce_mean(log_probs)

            loss = distortion + self.beta * rate

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.distortion_tracker.update_state(distortion)
        self.rate_tracker.update_state(rate)

        return {
            "loss": self.loss_tracker.result(),
            "distortion": self.distortion_tracker.result(),
            "rate": self.rate_tracker.result(),
        }

    def test_step(self, data):
        x = data
        outputs = self(x, training=False)
        recon = outputs["reconstruction"]
        z_indices = outputs["z_indices"]

        distortion = 1 - tf.reduce_mean(tf.image.ssim(
            tf.clip_by_value(x, 0, 1),
            tf.clip_by_value(recon, 0, 1),
            max_val=1.0
        ))

        log_probs = self.pixelcnn_prior.log_prob(z_indices)
        rate = -tf.reduce_mean(log_probs)
        loss = distortion + self.beta * rate

        self.loss_tracker.update_state(loss)
        self.distortion_tracker.update_state(distortion)
        self.rate_tracker.update_state(rate)

        return {
            "loss": self.loss_tracker.result(),
            "distortion": self.distortion_tracker.result(),
            "rate": self.rate_tracker.result(),
        }
    @property
    def tracked_metrics(self):
        return [self.loss_tracker, self.distortion_tracker, self.rate_tracker]
