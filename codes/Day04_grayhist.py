# %%
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
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


def plot_one_emotion_grayhist(data, img_arrays, img_labels, label=0):
    fig, axs = plt.subplots(2, 2, figsize=(25, 12))
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for i in range(0, 4, 2):
        idx = data[data['emotion'] == label].index[i]
        axs[i].imshow(img_arrays[idx][:, :, 0], cmap='gray')
        axs[i].set_title(emotions[img_labels[idx]])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

        axs[i+1].hist(img_arrays[idx][:, :, 0], 256, [0, 256])
        axs[i+1].set_title(emotions[img_labels[idx]])
        axs[i+1].set_xticklabels([])
        axs[i+1].set_yticklabels([])


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


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# %% 資料範例
df_raw = pd.read_csv("D:/mycodes/AIFER/data/fer2013.csv")
# %% 全類別圖片範例
df_train = df_raw[df_raw['Usage'] == 'Training']
X_train, y_train = prepare_data(df_train)
y_train_oh = to_categorical(y_train)
# %% 單一類別圖片範例
for label in emotions.keys():
    plot_one_emotion(df_train, X_train, y_train, label=label)
# %% 單一類別灰度直方圖
for label in emotions.keys():
    plot_one_emotion_grayhist(df_train, X_train, y_train, label=label)
# %% 灰度均值化
idx = 12
gray_img = X_train[idx][:, :, 0].astype("uint8")
gray_img = cv2.resize(gray_img, (600, 600))
plt.figure(figsize=(5, 5))
plt.imshow(gray_img, cmap='gray')
# %%
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
plt.plot(hist, 'r')
plt.show()
# %%
equalized_img = cv2.equalizeHist(gray_img)
cv2.imshow("image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
cv2.imshow("equal_image", equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
# plot image histogram
plt.hist(gray_img.ravel(), 256, [0, 255], label='original image')
plt.hist(equalized_img.ravel(), 256, [0, 255], label='equalized image')
plt.legend()
# %% CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2)
clahe_img = clahe.apply(gray_img)
cv2.imshow("clahe_img", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
# plot image histogram
plt.hist(gray_img.ravel(), 256, [0, 255], label='original image')
plt.hist(clahe_img.ravel(), 256, [0, 255], label='clahe image')
plt.legend()
# %%
