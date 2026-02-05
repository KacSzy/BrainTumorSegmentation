from tensorflow.keras import layers, Model, Input
from .blocks import conv_block, encoder_block, decoder_block


def build_unet_3d(input_shape, start_filters=32):
    """
    Builds a 3D U-Net model for volumetric segmentation.

    This architecture follows the U-Net design with an encoder-decoder structure
    connected by skip connections. It's optimized for 3D medical image segmentation
    tasks like brain tumor segmentation (BraTS).

    :param input_shape: Shape of the input volume (H, W, D, C) where C is the number of modalities.
    :param start_filters: Number of filters in the first encoder block. Doubles at each level. Defaults to 32.
    :return: Compiled Keras Model for 3D segmentation.
    """
    inputs = Input(input_shape)

    # --- ENCODER (Downsampling path) ---
    s1, p1 = encoder_block(inputs, start_filters)        # 128 -> 64
    s2, p2 = encoder_block(p1, start_filters * 2)        # 64 -> 32
    s3, p3 = encoder_block(p2, start_filters * 4)        # 32 -> 16
    s4, p4 = encoder_block(p3, start_filters * 8)        # 16 -> 8

    # --- BRIDGE (Deepest abstract features) ---
    b1 = conv_block(p4, start_filters * 16)              # 8x8x8

    # --- DECODER (Upsampling path - reconstructs segmentation mask) ---
    d1 = decoder_block(b1, s4, start_filters * 8)        # 8 -> 16
    d2 = decoder_block(d1, s3, start_filters * 4)        # 16 -> 32
    d3 = decoder_block(d2, s2, start_filters * 2)        # 32 -> 64
    d4 = decoder_block(d3, s1, start_filters)            # 64 -> 128

    # --- OUTPUT (Per-voxel classification) ---
    outputs = layers.Conv3D(4, (1, 1, 1), activation="softmax")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net_3D")
    return model
