from constriction.stream import QueueEncoder
from constriction.stream import QueueDecoder
from constriction.symbol import Categorical
import tensorflow as tf

# ---------------------------
# Compression: Latent to Bitstream
# ---------------------------
def encode_latents_aac(prediction):
    z_q = prediction["quantized"]
    z_indices = prediction["z_indices"]
    z_flat = tf.reshape(z_indices, [-1])           # shape (batch * T * H * W,)
    probs = tf.nn.softmax(z_q, axis=-1)
    probs_flat = tf.reshape(probs, [-1, probs.shape[-1]])  # shape (batch * T * H * W, num_embeddings)
    encoder = QueueEncoder()

    for i in range(z_flat.shape[0]):
        pmf = probs_flat[i].numpy()  # get probability vector for this symbol
        model = Categorical(pmf)
        symbol = int(z_flat[i].numpy())
        encoder.encode_symbol(symbol, model)
    compressed_bitstream = encoder.get_compressed()  # returns np.array of uint32
    return compressed_bitstream


# ================================
# Decompression: Decode from a bitstream
# ================================
def decompress_video(bitstream, vq_layer, pixelcnn_prior, original_shape):
    batch_size, T, H, W = original_shape
    decoder_stream = QueueDecoder(bitstream)
    z_decoded = tf.zeros((batch_size, T, H, W), dtype=tf.int32)

    for idx in range(batch_size * T * H * W):
        # Reshape z_decoded to partial conditioning
        z_partial = tf.reshape(z_decoded, (batch_size, T, H, W))

        probs = pixelcnn_prior.prob(z_partial)
        probs_flat = tf.reshape(probs, [-1, probs.shape[-1]])

        # Decode next symbol
        model = Categorical(probs_flat[idx].numpy())
        symbol = decoder_stream.decode_symbol(model)

        # Update z_decoded
        z_decoded = tf.tensor_scatter_nd_update(
            z_decoded,
            indices=[[idx]],
            updates=[symbol]
        )

    # Reconstruct z_q from bitstream
    z_decoded = tf.reshape(z_decoded, (batch_size, T, H, W))
    z_onehot = tf.one_hot(z_decoded, depth=vq_layer.num_embeddings)

    embeddings = tf.transpose(vq_layer.embeddings, [1, 0]) 
    z_quantized = tf.tensordot(z_onehot, embeddings, axes=[[4], [0]])
    return z_quantized # pass z_quantized through the decoder portion of the vqvae to get video
