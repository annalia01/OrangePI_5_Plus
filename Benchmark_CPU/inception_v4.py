from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

# URL pesi preaddestrati
TF_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_tf_kernels.h5"

def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

def inception_stem(input):
    channel_axis = -1

    x = conv_block(input, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, (3, 3), strides=(2, 2), padding='valid')
    x = Concatenate(axis=channel_axis)([x1, x2])

    x1 = conv_block(x, 64, (1, 1))
    x1 = conv_block(x1, 96, (3, 3), padding='valid')

    x2 = conv_block(x, 64, (1, 1))
    x2 = conv_block(x2, 64, (1, 7))
    x2 = conv_block(x2, 64, (7, 1))
    x2 = conv_block(x2, 96, (3, 3), padding='valid')

    x = Concatenate(axis=channel_axis)([x1, x2])

    x1 = conv_block(x, 192, (3, 3), strides=(2, 2), padding='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=channel_axis)([x1, x2])
    return x

def inception_A(input):
    channel_axis = -1

    a1 = conv_block(input, 96, (1, 1))

    a2 = conv_block(input, 64, (1, 1))
    a2 = conv_block(a2, 96, (3, 3))

    a3 = conv_block(input, 64, (1, 1))
    a3 = conv_block(a3, 96, (3, 3))
    a3 = conv_block(a3, 96, (3, 3))

    a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, (1, 1))

    return Concatenate(axis=channel_axis)([a1, a2, a3, a4])

def inception_B(input):
    channel_axis = -1

    b1 = conv_block(input, 384, (1, 1))

    b2 = conv_block(input, 192, (1, 1))
    b2 = conv_block(b2, 224, (1, 7))
    b2 = conv_block(b2, 256, (7, 1))

    b3 = conv_block(input, 192, (1, 1))
    b3 = conv_block(b3, 192, (7, 1))
    b3 = conv_block(b3, 224, (1, 7))
    b3 = conv_block(b3, 224, (7, 1))
    b3 = conv_block(b3, 256, (1, 7))

    b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, (1, 1))

    return Concatenate(axis=channel_axis)([b1, b2, b3, b4])

def inception_C(input):
    channel_axis = -1

    c1 = conv_block(input, 256, (1, 1))

    c2 = conv_block(input, 384, (1, 1))
    c2_1 = conv_block(c2, 256, (1, 3))
    c2_2 = conv_block(c2, 256, (3, 1))
    c2 = Concatenate(axis=channel_axis)([c2_1, c2_2])

    c3 = conv_block(input, 384, (1, 1))
    c3 = conv_block(c3, 448, (3, 1))
    c3 = conv_block(c3, 512, (1, 3))
    c3_1 = conv_block(c3, 256, (1, 3))
    c3_2 = conv_block(c3, 256, (3, 1))
    c3 = Concatenate(axis=channel_axis)([c3_1, c3_2])

    c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, (1, 1))

    return Concatenate(axis=channel_axis)([c1, c2, c3, c4])

def reduction_A(input):
    channel_axis = -1

    r1 = conv_block(input, 384, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input, 192, (1, 1))
    r2 = conv_block(r2, 224, (3, 3))
    r2 = conv_block(r2, 256, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    return Concatenate(axis=channel_axis)([r1, r2, r3])

def reduction_B(input):
    channel_axis = -1

    r1 = conv_block(input, 192, (1, 1))
    r1 = conv_block(r1, 192, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input, 256, (1, 1))
    r2 = conv_block(r2, 256, (1, 7))
    r2 = conv_block(r2, 320, (7, 1))
    r2 = conv_block(r2, 320, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    return Concatenate(axis=channel_axis)([r1, r2, r3])

def create_inception_v4(nb_classes=1001, load_weights=True):
    input_layer = Input((299, 299, 3))
    x = inception_stem(input_layer)

    for _ in range(4):
        x = inception_A(x)

    x = reduction_A(x)

    for _ in range(7):
        x = inception_B(x)

    x = reduction_B(x)

    for _ in range(3):
        x = inception_C(x)

    x = AveragePooling2D((8, 8))(x)
    x = Dropout(0.8)(x)
    x = Flatten()(x)
    out = Dense(units=nb_classes, activation='softmax')(x)

    model = Model(input_layer, out, name='Inception-v4')

    if load_weights:
        weights = get_file('inception_v4_weights_tf_dim_ordering_tf_kernels.h5',
                           TF_BACKEND_TF_DIM_ORDERING, cache_subdir='models')
        model.load_weights(weights)
        print("Model weights loaded.")

    return model

if __name__ == "_main_":
    model = create_inception_v4(load_weights=True)
    model.summary()
