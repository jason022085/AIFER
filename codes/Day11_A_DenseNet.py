import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Concatenate, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D


def DenseLayer(x, growthRate, dropRate=0):

    # bottleneck
    x = BatchNormalization(axis=3)(x)
    x = ReLU(x)
    x = Conv2D(4*growthRate, kernel_size=(1, 1), padding='same')(x)

    # composition
    x = BatchNormalization(axis=3)(x)
    x = ReLU(x)
    x = Conv2D(growthRate, kernel_size=(3, 3), padding='same')(x)

    # dropout
    x = Dropout(dropRate)(x)

    return x


def DenseBlock(x, num_layers, growthRate, dropRate=0):

    for i in range(num_layers):
        featureMap = DenseLayer(x, growthRate, dropRate)
        x = Concatenate([x, featureMap], axis=3)

    return x


def TransitionLayer(x, ratio):

    growthRate = int(x.shape[-1]*ratio)
    x = BatchNormalization(axis=3)(x)
    x = ReLU(x)
    x = Conv2D(growthRate, kernel_size=(1, 1),
               strides=(2, 2), padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x


def DenseNet121(numClass=1000, inputShape=(224, 224, 3), growthRate=12):
    x_in = Input(inputShape)
    x = Conv2D(growthRate*2, (3, 3), padding='same')(x_in)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = TransitionLayer(DenseBlock(
        x, num_layers=6, growthRate=12, dropRate=0.2))
    x = TransitionLayer(DenseBlock(
        x, num_layers=12, growthRate=12, dropRate=0.2))
    x = TransitionLayer(DenseBlock(
        x, num_layers=24, growthRate=12, dropRate=0.2))
    x = DenseBlock(x, num_layers=16, growthRate=12, dropRate=0.2)
    x = GlobalAveragePooling2D()(x)
    x_out = Dense(numClass=1000, activation='softmax')(x)

    model = Model(x_in, x_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model
