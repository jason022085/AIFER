# %%
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import random_rotation, random_shear, random_zoom, random_brightness
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os
import itertools
import mlflow.tensorflow
import mlflow
import cv2
# %%
radam = tfa.optimizers.RectifiedAdam(lr=0.001)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
sgd = tf.keras.optimizers.SGD(
    lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
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


def resize_image(img_array, output_shape=(224, 224)):
    output_img = cv2.resize(img_array, output_shape)
    return output_img


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


def build_model(preModel=EfficientNetB0, pretrained=True, optimizer="SGD", num_classes=7, input_shape=(48, 48, 3), l2_coef=0.001):

    pre_model = preModel(include_top=False, weights='imagenet' if pretrained == True else None,
                         input_shape=input_shape,
                         pooling='max', classifier_activation='softmax')

    for layer in pre_model.layers:
        layer.trainable = False

    x = Dropout(0.2)(pre_model.output)

    output = Dense(
        num_classes, activation="softmax", name="main_output",
        kernel_regularizer=regularizers.l2(l2_coef))(x)

    freezed_model = tf.keras.Model(pre_model.input, output)

    if optimizer == "SGD":
        opt = sgd
    elif optimizer == "ADAM":
        opt = adam
    elif optimizer == "RANGER":
        opt = ranger

    freezed_model.compile(optimizer=opt,
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['accuracy', 'MeanSquaredError', 'AUC'])

    return freezed_model


def step_decay(epoch):
    """
    Step decay schedule drops the learning rate by a factor every few epochs.
    Restart at every 10 epoch
    """
    lr_init = 0.001
    drop = 0.5
    global_step = 1+epoch
    cycle_step = 25
    drop_epoch = 5
    if global_step <= 50:
        if global_step >= cycle_step:
            global_step = global_step % cycle_step
        # 每epochs_drop個epoch，lr乘以drop倍。
        lr = lr_init * (drop**(int(((global_step)/drop_epoch))))
    else:
        lr = lr_init * (drop**(int(((global_step)/drop_epoch))))
    return float(lr)


def exp_decay(epoch):

    lr_init = 0.001
    global_step = 1+epoch
    cycle_step = 25

    if global_step <= 50:
        if global_step >= cycle_step:
            global_step = global_step % cycle_step
        lr = lr_init * tf.math.exp(-0.2 * (global_step))
    else:
        lr = lr_init * tf.math.exp(-0.2 * (global_step))
    return float(lr)


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


def unfreeze_model(model, optimizer="SGD", n=1, is_unfreeze_BN=False):
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

    if optimizer == "SGD":
        opt = sgd
    elif optimizer == "ADAM":
        opt = adam
    elif optimizer == "RANGER":
        opt = ranger

    model.compile(optimizer=opt,
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

X_train_all_norm = X_train_all/255  # 如果要加速收斂，要先正規畫
X_val_norm = X_val/255
# %% 調整每類權重
is_aug = True
class_sample = y_train if is_aug == False else y_train_all
class_sample_size = [np.where(class_sample == c)[0].shape[0]
                     for c in range(len(emotions.keys()))]
max_class_size = np.max(class_sample_size)
class_weight = [max_class_size/size for size in class_sample_size]
class_weight = dict(zip(emotions.keys(), class_weight))
custom_weight = dict(zip(emotions.keys(), [10, 1, 10, 1, 10, 1, 10]))
# %% 1. 單階段訓練模型 with X_train_all
RLOP = ReduceLROnPlateau(monitor='val_loss',
                         factor=0.5, patience=5, verbose=1)
POLY_DECAY = LearningRateScheduler(poly_decay)
STEP_DECAY = LearningRateScheduler(step_decay)
EXPO_DECAY = LearningRateScheduler(exp_decay)
optimizer_space = ["ADAM"]  # "ADAM","SGD","RANGER"
unfreeze_space = [1000]  # 6, 19, 78, 1000
epochs = 100
batch_size = 128
"""
    callback_space = {"POLY": [POLY_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_POLY_aug")],
                      "STEP": [STEP_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_STEP_aug")],
                      "EXPO": [EXPO_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_EXPO_aug")],
                      "RLOP": [RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_RLOP_aug")],
                      "NONE": [ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_aug")]}


    class_weight_space = {"balance": class_weight,
                      "custom": custom_weight}
"""
for opt in optimizer_space:
    for unfreeze_n in unfreeze_space:
        class_weight_space = {"custom": custom_weight}
        for cw_name, cw in class_weight_space.items():
            callback_space = {"RLOP": [RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_RLOP_aug")],}
            for decay_name, cb in callback_space.items():
                model = build_model(optimizer=opt)
                unfreeze_model(model, optimizer=opt,
                               n=unfreeze_n, is_unfreeze_BN=True)
                with mlflow.start_run(experiment_id=2, run_name=f"EFNB0_{opt}_free{unfreeze_n}_{decay_name}_CW{cw_name}_aug"):
                    mlflow.tensorflow.autolog()
                    hist = model.fit(X_train_all, y_train_all_oh,
                                     validation_data=(X_val, y_val_oh),
                                     initial_epoch=0,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     class_weight=cw,
                                     callbacks=cb)
                mlflow.end_run()

# %% 2. 單階段訓練模型 with X_train_all but not pretrained
# 理由: 資料量夠多其實可以不用預訓練權重
RLOP = ReduceLROnPlateau(monitor='val_loss',
                         factor=0.75, patience=5, verbose=1)
POLY_DECAY = LearningRateScheduler(poly_decay)
STEP_DECAY = LearningRateScheduler(step_decay)
EXPO_DECAY = LearningRateScheduler(exp_decay)
optimizer_space = ["ADAM"]  # "ADAM","SGD","RANGER"
unfreeze_space = [1000]  # 6, 19, 78, 1000
epochs = 100
batch_size = 128
"""
    callback_space = {"POLY": [POLY_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_POLY_aug")],
                      "STEP": [STEP_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_STEP_aug")],
                      "EXPO": [EXPO_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_EXPO_aug")],
                      "RLOP": [RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_RLOP_aug")],
                      "NONE": [ckpt(test_name=f"{opt}_free{unfreeze_n}_cw{cw_name}_aug")]}


    class_weight_space = {"balance": class_weight,
                      "custom": custom_weight}
"""
for opt in optimizer_space:
    for unfreeze_n in unfreeze_space:
        class_weight_space = {"balance": class_weight,
                              "custom": custom_weight}
        for cw_name, cw in class_weight_space.items():
            callback_space = {"POLY": [POLY_DECAY, RLOP, ckpt(
                test_name=f"{opt}_free{unfreeze_n}_CW{cw_name}_POLY_aug")]}
            for decay_name, cb in callback_space.items():
                model = build_model(optimizer=opt, pretrained=False)
                unfreeze_model(model, optimizer=opt,
                               n=unfreeze_n, is_unfreeze_BN=True)
                with mlflow.start_run(experiment_id=2, run_name=f"EFNB0_init_{opt}_free{unfreeze_n}_{decay_name}_CW{cw_name}_aug"):
                    mlflow.tensorflow.autolog()

                    hist = model.fit(X_train_all_norm, y_train_all_oh,
                                     validation_data=(X_val_norm, y_val_oh),
                                     initial_epoch=0,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     class_weight=cw,
                                     callbacks=cb)
                mlflow.end_run()
# %% 3. 單階段訓練模型 with X_train
RLOP = ReduceLROnPlateau(monitor='val_loss',
                         factor=0.75, patience=5, verbose=1)
POLY_DECAY = LearningRateScheduler(poly_decay)
optimizer_space = ["ADAM", "RANGER"]  # "ADAM","SGD","RANGER"
unfreeze_space = [1000]  # 6, 19, 78, 1000
epochs = 100
batch_size = 128
"""
    callback_space = {"POLY": [POLY_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_POLY")],
                      "RLOP": [RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_RLOP")],
                      "NONE": [ckpt(test_name=f"{opt}_free{unfreeze_n}")]}
"""
for opt in optimizer_space:
    for unfreeze_n in unfreeze_space:
        callback_space = {"POLY": [POLY_DECAY, RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_POLY")],
                          "RLOP": [RLOP, ckpt(test_name=f"{opt}_free{unfreeze_n}_RLOP")],
                          "NONE": [ckpt(test_name=f"{opt}_free{unfreeze_n}")]}
        for decay_name, cb in callback_space.items():
            model = build_model(optimizer=opt)
            unfreeze_model(model, optimizer=opt,
                           n=unfreeze_n, is_unfreeze_BN=True)
            with mlflow.start_run(experiment_id=2, run_name=f"EFNB0_{opt}_free{unfreeze_n}_{decay_name}"):
                mlflow.tensorflow.autolog()
                hist = model.fit(X_train, y_train_oh,
                                 validation_data=(X_val, y_val_oh),
                                 initial_epoch=0,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 class_weight=class_weight,
                                 callbacks=cb)
            mlflow.end_run()

# %% 4. 多階段式訓練 with X_train_all(微調 block 7 layers -> block 6 layers -> all layers)
# 有BUG
# 因為耗時太久，所以只選上面最好的模型出來改用多階段訓練。

unfreeze_n = [19, 78, 1000]
phase_epochs = [30, 60, 100]
batch_size = 128
optimizer_space = ["SGD", "ADAM"]
phases = len(phase_epochs)
# 開始訓練

for i in range(phases):
    unfreeze_model(model, optimizer=opt,
                   n=unfreeze_n[i], is_unfreeze_BN=True)
    with mlflow.start_run(experiment_id=2, run_name=f"EFNB0_{opt}_{decay_name}_aug_phase{i+1}"):

        mlflow.tensorflow.autolog()
        hist = model.fit(X_train_all, y_train_all_oh,
                         validation_data=(X_val, y_val_oh),
                         initial_epoch=0 if i == 0 else phase_epochs[i-1],
                         epochs=phase_epochs[i],
                         batch_size=batch_size,
                         class_weight=class_weight,
                         callbacks=cb)
        phase_history[i] = hist
    mlflow.end_run()
