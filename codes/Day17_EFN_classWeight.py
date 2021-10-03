# %%
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import random_rotation, random_shear, random_zoom
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import itertools
import mlflow.tensorflow
import mlflow
import cv2
# %%


def prepare_data(data, to_3_channels=True, to_clahe=False):
    """ Prepare data for modeling
        input: data frame with labels and pixel data
        output: image and label array in shape(48,48,3) and pixel range(0,256) """
    clahe = cv2.createCLAHE(clipLimit=2)
    channels = 3 if to_3_channels == True else 1

    image_array = np.zeros(shape=(len(data), 48, 48, channels))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))  # 灰階圖的channel數為1

        #  CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if to_clahe == True:
            image = image[:, :, 0].astype("uint8")
            image = clahe.apply(image)
            image = np.reshape(image, (48, 48, 1))

        # Convert to 3 channels
        if to_3_channels == True:
            image = np.stack(
                [image[:, :, 0], image[:, :, 0], image[:, :, 0]], axis=-1)
        image_processed = preprocess_input(image)
        image_array[i] = image_processed

    return image_array, image_label


def build_model(preModel=EfficientNetB0, num_classes=7):

    pre_model = preModel(include_top=False, weights='imagenet',
                         input_shape=(48, 48, 3),
                         pooling='max', classifier_activation='softmax')

    output_layer = Dense(
        num_classes, activation="softmax", name="main_output")

    model = tf.keras.Model(
        pre_model.input, output_layer(pre_model.output))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model


def resize_image(img_array, output_shape=(224, 224)):
    output_img = cv2.resize(img_array, output_shape)
    return output_img


def augmentation_image(img_array):
    tf.random.set_seed(19960220)
    img_array = random_rotation(img_array, rg=30, channel_axis=2)  # 旋轉
    img_array = random_shear(img_array, intensity=20, channel_axis=2)  # 剪裁
    img_array = random_zoom(img_array, zoom_range=(
        0.8, 0.8), channel_axis=2)  # 縮放
    return img_array


def auto_augmentation(X_train, y_train, class_sample_size, ratio=1):
    max_class_size = np.max(class_sample_size)
    fill_class_sample_size = [int(ratio*max_class_size - size)
                              for size in class_sample_size]
    X_train_aug_array = []
    y_train_aug_array = []
    for i, fill_size in enumerate(fill_class_sample_size):
        samples = np.random.choice(list(np.where(y_train == i)[0]), fill_size)
        for image in X_train[samples]:
            image_aug = augmentation_image(image)
            X_train_aug_array.append(image_aug)
            y_train_aug_array.append(i)
    X_train_aug_array = np.array(X_train_aug_array)
    y_train_aug_array = np.array(y_train_aug_array)
    return X_train_aug_array, y_train_aug_array


def plot_one_emotion(data, img_arrays, img_labels, label=0):
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for i in range(5):
        idx = data[data['emotion'] == label].index[i]
        axs[i].imshow(img_arrays[idx][:, :, 0], cmap='gray')
        axs[i].set_title(emotions[img_labels[idx]])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])


def step_decay(epoch):
    """
    Warm-up applying high learning rate at first few epochs.
    Step decay schedule drops the learning rate by a factor every few epochs.
    """
    lr_init = 0.003
    drop = 0.5
    epochs_drop = 5
    warm_up_epoch = 5
    if epoch <= warm_up_epoch:
        lr = (epoch+1) / warm_up_epoch
    else:
        lr = lr_init * (drop**(int(((1+epoch)/epochs_drop))))
    return lr


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# %% 資料讀取
df_raw = pd.read_csv("D:/mycodes/AIFER/data/FER2013/fer2013.csv")
#  資料前處理(CLAHE)
X_train, y_train = prepare_data(df_raw[df_raw['Usage'] == 'Training'])
X_val, y_val = prepare_data(df_raw[df_raw['Usage'] == 'PublicTest'])
# %% 調整每類權重
class_sample_size = [np.where(y_train == c)[0].shape[0]
                     for c in range(len(emotions.keys()))]
max_class_size = np.max(class_sample_size)
class_weight = [max_class_size/size for size in class_sample_size]
class_weight = dict(zip(emotions.keys(), class_weight))
# %% 建立預訓練模型
model = build_model()
prob_res = model(X_train[:1]).numpy()
print(f"EFN build successfully!")

# %% 訓練

epochs = 30
batch_size = 32
model = build_model()
y_train_oh, y_val_oh = to_categorical(y_train), to_categorical(y_val)
with mlflow.start_run(experiment_id=1, run_name="Class weight"):
    mlflow.tensorflow.autolog()
    hist1 = model.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),
                      epochs=epochs, batch_size=batch_size, class_weight=class_weight)
    # ,callbacks = callbacks_list)
mlflow.end_run()
# %%
