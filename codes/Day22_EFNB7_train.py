# %%
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import random_rotation, random_shear, random_zoom, random_brightness
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
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
adam = tf.keras.optimizers.Adam(lr=0.001)


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


def augmentation_image(x):
    tf.random.set_seed(19960220)
    x = random_rotation(x, rg=15, channel_axis=2)  # 旋轉 0~15度
    x = random_brightness(x, (1.1, 1.3))  # 調整明度(1.1~1.3倍)
    x = random_shear(x, intensity=10, channel_axis=2)  # 錯切
    x = random_zoom(x, zoom_range=(0.8, 1), channel_axis=2)  # 縮放(1.00 ~ 1.25倍)
    return x


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


def build_model(preModel=EfficientNetB7,  num_classes=7, input_shape=(48, 48, 3), l2_coef=0.0001):

    pre_model = preModel(include_top=False, weights='imagenet',
                         input_shape=input_shape,
                         pooling='max', classifier_activation='softmax')

    for layer in pre_model.layers:
        layer.trainable = False

    x = Dropout(0.2)(pre_model.output)

    output = Dense(
        num_classes, activation="softmax", name="main_output",
        kernel_regularizer=regularizers.l2(l2_coef))(x)

    freezed_model = tf.keras.Model(pre_model.input, output)
    freezed_model.compile(optimizer=adam,
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['accuracy', 'MeanSquaredError', 'AUC'])

    return freezed_model


def poly_decay(epoch):

    lr_init = 0.001
    lr_end = 0.00001
    decay_step = 25
    global_step = 1+epoch
    power = 0.5

    if global_step <= 50:
        if global_step >= decay_step:
            global_step = global_step % decay_step

        lr = (lr_init-lr_end)*((1-(global_step/decay_step))**power) + lr_end
    else:
        decay_step = 100
        lr = (lr_init-lr_end)*((1-(global_step/decay_step))**power) + lr_end

    return float(lr)


def unfreeze_model(model, lr=0.001, n=1000, is_unfreeze_BN=False):
    # We unfreeze the top n layers while leaving BatchNorm layers frozen
    # n = 6 (~ block-top)
    # n = 19 (~ block-7)
    if is_unfreeze_BN == False:
        for layer in model.layers[-n:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
    else:
        for layer in model.layers[-n:]:
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'MeanSquaredError', 'AUC'])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def ckpt(test_name="test"):
    if not os.path.exists("./checkpoints/"+test_name):
        os.makedirs("./checkpoints/"+test_name)
    checkpoint = ModelCheckpoint("./checkpoints/"+test_name+"/"+test_name+"_epoch{epoch:02d}_valacc{val_accuracy:.4f}.h5",
                                 monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='max')
    return checkpoint


def save_model(model, save_folder=".", save_name="mymodel"):
    # 儲存模型-tf格式
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model.
    with open(save_folder + "/" + save_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

    # 儲存模型-keras格式
    model.save(save_folder + "/" + save_name + ".h5")


def evaluate_model(model, X_val, y_val):
    # 混淆矩陣
    y_prob_val = model.predict(X_val)
    y_pred_val = np.argmax(y_prob_val, axis=1)
    classification_report(
        y_val, y_pred_val, labels=emotions.values(), digits=4)
    plot_confusion_matrix(confusion_matrix(y_val, y_pred_val),
                          classes=list(emotions.values()))


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# %% main
#  資料讀取
df_raw = pd.read_csv("D:/mycodes/AIFER/data/FER2013/fer2013.csv")

#  資料前處理
X_train, y_train = prepare_data(df_raw[df_raw['Usage'] == 'Training'])
X_val, y_val = prepare_data(df_raw[df_raw['Usage'] == 'PublicTest'])
y_train_oh,  y_val_oh = to_categorical(y_train), to_categorical(y_val)

# 資料擴增至每類都有最大類的資料量
# class_sample_size = [np.where(y_train == c)[0].shape[0]
#                      for c in range(len(emotions.keys()))]

# X_train_aug, y_train_aug = auto_augmentation(
#     X_train, y_train, class_sample_size, ratio=1)

# X_train_all = np.vstack((X_train, X_train_aug))
# y_train_all = np.hstack((y_train, y_train_aug))

# 讀取已經增強的平衡資料集
data_train_all = np.load("./data/data_train_with_aug.npz")
X_train_all, y_train_all = data_train_all['X'], data_train_all['y']
y_train_all_oh = to_categorical(y_train_all)
# %% 調整每類權重
class_sample_size = [np.where(y_train_all == c)[0].shape[0]
                     for c in range(len(emotions.keys()))]
max_class_size = np.max(class_sample_size)
class_weight = [max_class_size/size for size in class_sample_size]
class_weight = dict(zip(emotions.keys(), class_weight))
# %% 最終模型訓練 with X_train_all
RLOP = ReduceLROnPlateau(monitor='val_accuracy',
                         factor=0.75, patience=5, verbose=1)
POLY_DECAY = LearningRateScheduler(poly_decay)
epochs = 100
batch_size = 32
cw = class_weight
cb = [POLY_DECAY, RLOP, ckpt(test_name=f"ADAM_FREE1000_balanceCW_POLY_AUG")]

model = build_model()
unfreeze_model(model, n=1000, is_unfreeze_BN=True)
with mlflow.start_run(experiment_id=2, run_name=f"EFNB7_ADAM_FREE1000_balanceCW_POLY_AUG"):
    mlflow.tensorflow.autolog()
    hist = model.fit(X_train_all, y_train_all_oh,
                     validation_data=(X_val, y_val_oh),
                     initial_epoch=0,
                     epochs=epochs,
                     batch_size=batch_size,
                     class_weight=cw,
                     callbacks=cb)
mlflow.end_run()
# %% 最後用極小的、固定的lr訓練10輪
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

with mlflow.start_run(experiment_id=2, run_name=f"FINAL_EFNB7_ADAM_FREE1000_balanceCW_CONST_AUG"):
    mlflow.tensorflow.autolog()
    hist = model.fit(X_train_all, y_train_all_oh,
                     validation_data=(X_val, y_val_oh),
                     initial_epoch=100,
                     epochs=110,
                     batch_size=32,
                     class_weight=cw,
                     callbacks=[ckpt(test_name=f"FINAL_ADAM_FREE1000_balanceCW_CONST_AUG")])
mlflow.end_run()
# %% 二階段式訓練(微調 last layer -> all layers) with X_train_all
"""
unfreeze_n = [1, 1000]
phase_initLR = [0.001, 0.00001]
phase_epochs = [30, 100]
phase_batch_size = [32, 32]
phases = len(phase_epochs)
phase_cb = [[POLY_DECAY, RLOP, ckpt(test_name=f"ADAM_FREE1_balanceCW_POLY_AUG_p1")],
            [RLOP, ckpt(test_name=f"ADAM_FREE1000_balanceCW_RLOP_AUG_p2")]]
model = build_model()

for i in range(phases):
    unfreeze_model(model, lr=phase_initLR[i],
                   n=unfreeze_n[i], is_unfreeze_BN=False)
    with mlflow.start_run(experiment_id=2, run_name=f"EFNB7_ADAM_balanceCW_AUG_p{i+1}"):

        mlflow.tensorflow.autolog()
        hist = model.fit(X_train_all, y_train_all_oh,
                         validation_data=(X_val, y_val_oh),
                         initial_epoch=0 if i == 0 else phase_epochs[i-1],
                         epochs=phase_epochs[i],
                         batch_size=phase_batch_size[i],
                         class_weight=cw,
                         callbacks=cb)
    mlflow.end_run()
save_model(model, "effB7_final_2Steps")
"""
# %%
