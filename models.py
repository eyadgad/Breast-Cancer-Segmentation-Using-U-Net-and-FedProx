# import required libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Dropout


# === Helper Functions for Model Creation === #
def create_upsample_layer(filters, kernel_size, padding='same', kernel_initializer='he_normal'):
    """
    Creates an upsampling layer with Conv2D, BatchNormalization, ReLU, and UpSampling2D.
    """
    def layer(x):
        x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D(size=(2, 2))(x)
        return x
    return layer


def create_downsample_layer(filters, kernel_size, padding='same', kernel_initializer='he_normal'):
    """
    Creates a downsampling layer with Conv2D, BatchNormalization, ReLU, and MaxPooling2D.
    """
    def layer(x):
        x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        return x
    return layer


def create_model():
    """
    Creates the DeepUNet architecture with downsampling and upsampling layers.
    """
    dropout_rate = 0.2
    inputs = Input(shape=(256, 256, 3))

    # Encoder
    x = create_downsample_layer(64, 3)(inputs)
    x = create_downsample_layer(128, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_downsample_layer(256, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_downsample_layer(512, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_downsample_layer(1024, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_downsample_layer(2048, 3)(x)
    x = Dropout(dropout_rate)(x)

    # Decoder
    x = create_upsample_layer(1024, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_upsample_layer(512, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_upsample_layer(256, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_upsample_layer(128, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_upsample_layer(64, 3)(x)
    x = Dropout(dropout_rate)(x)
    x = create_upsample_layer(32, 3)(x)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model




