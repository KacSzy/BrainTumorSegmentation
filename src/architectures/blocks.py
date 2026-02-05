import tensorflow as tf
from tensorflow.keras import layers


def conv_block(input_tensor, num_filters):
    """
    Creates a convolutional block with two 3D convolution layers.

    This block performs two consecutive 3D convolutions, each followed by
    batch normalization and ReLU activation.

    :param input_tensor: Input tensor to the convolutional block.
    :param num_filters: Number of filters for the convolution layers.
    :return: Output tensor after applying two Conv3D-BatchNorm-ReLU sequences.
    """
    # First convolution sequence
    x = layers.Conv3D(filters=num_filters, kernel_size=(3, 3, 3), padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Second convolution sequence
    x = layers.Conv3D(filters=num_filters, kernel_size=(3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(input_tensor, num_filters):
    """
    Creates an encoder block for U-Net architecture.

    This block applies a convolutional block followed by max pooling for downsampling.
    Returns both the convolutional output (for skip connections) and the pooled output.

    :param input_tensor: Input tensor to the encoder block.
    :param num_filters: Number of filters for the convolution layers.
    :return: Tuple of (conv_output, pooled_output) where conv_output is used for skip connections
             and pooled_output is passed to the next encoder level.
    """
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    return x, p


def decoder_block(input_tensor, skip_features, num_filters):
    """
    Creates a decoder block for U-Net architecture.

    This block performs upsampling via transposed convolution, concatenates with
    skip connection features from the encoder, and applies a convolutional block.

    :param input_tensor: Input tensor from the previous decoder level.
    :param skip_features: Feature map from the corresponding encoder level (skip connection).
    :param num_filters: Number of filters for the convolution layers.
    :return: Output tensor after upsampling, concatenation, and convolution.
    """

    # Upsample using transposed convolution
    x = layers.Conv3DTranspose(filters=num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(
        input_tensor)

    # Concatenate with skip connection features
    x = layers.concatenate([x, skip_features])

    # Apply convolutional block
    x = conv_block(x, num_filters)
    return x
