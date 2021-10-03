# %%

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import efficientnet.tfkeras as efn
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import mlflow.tensorflow
import mlflow
# %%


def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))  # 灰階圖的channel數為1
        image_array[i] = image

    return image_array, image_label


def convert_to_3_channels(img_arrays):
    sample_size, nrows, ncols, c = img_arrays.shape
    img_stack_arrays = np.zeros((sample_size, nrows, ncols, 3))
    for _ in range(sample_size):
        img_stack = np.stack(
            [img_arrays[_][:, :, 0], img_arrays[_][:, :, 0], img_arrays[_][:, :, 0]], axis=-1)
        img_stack_arrays[_] = img_stack/255
    return img_stack_arrays


def build_model(preModel=efn.EfficientNetB0, num_classes=7):

    pred_model = preModel(include_top=False, weights='imagenet',
                          input_shape=(48, 48, 3),
                          pooling='max')
    output_layer = Dense(
        num_classes, activation="softmax", name="output_layer")

    model = tf.keras.Model(
        pred_model.inputs, output_layer(pred_model.output))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# %% 資料讀取
df_raw = pd.read_csv("D:/mycodes/AIFER/data/fer2013.csv")
# 資料切割(訓練、驗證、測試)
X_train, y_train = prepare_data(df_raw[df_raw['Usage'] == 'Training'])
X_val, y_val = prepare_data(df_raw[df_raw['Usage'] == 'PublicTest'])
X_train, X_val = convert_to_3_channels(X_train), convert_to_3_channels(X_val)
y_train_oh, y_val_oh = to_categorical(y_train), to_categorical(y_val)

# %% 建立預訓練模型
# 測試模型是否建立成功

model = build_model()
prob_res = model(X_train[:1]).numpy()
print(f"EfficientNetB0 build successfully!")

# %% 針對每種預訓練模型去訓練
epochs = 30
batch_size = 32
preModelDict = {"efn.EfficientNetB0": efn.EfficientNetB0}
for k, v in preModelDict.items():
    model = build_model(preModel=v)
    with mlflow.start_run(experiment_id=0, run_name=k):
        mlflow.tensorflow.autolog()
        hist1 = model.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),
                          epochs=epochs, batch_size=batch_size)
    mlflow.end_run()
# %%
