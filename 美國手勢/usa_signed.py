import pandas as pd
train_df = pd.read_csv("sign_mnist_train.csv")
valid_df = pd.read_csv("sign_mnist_test.csv")
train_df.head()
train_df.info()

#label即為目標值
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']
X_train = train_df.values
X_valid = valid_df.values
X_train.shape
y_train.shape
y_train[0]

import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

num_images = 20
for i in range(num_images):
    row = X_train[i]
    label = y_train[i]
    image = row.reshape(28,28)
    plt.subplot(1, num_images,i+1)
    plt.title(letters[label], fontdict={'fontsize':50, 'color': 'red'})
    plt.axis('off')
    plt.imshow(image, cmap='gray')

X_train = X_train / 255
X_valid = X_valid / 255


import tensorflow.keras as keras
num_classes = 26
y_train = keras.utils.to_categorical(y_train,num_classes)
y_valid = keras.utils.to_categorical(y_valid,num_classes)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, activation='relu',input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_valid, y_valid))

model.save('basic-nn-model')