import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D, Dropout, MaxPool2D
import sklearn
from sklearn.model_selection import train_test_split

lines = []
data = "./data/driving_log.csv"

with open(data) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction_right = 0.1
correction_left = 0.1

for line in lines:
    for i in range(1):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename

        image = cv2.imread(current_path)
        images.append(image)
        if(i == 0):
            images.append(np.fliplr(image))

        measurement = float(line[3])
        if(i == 0):
            measurements.append(measurement)
            measurements.append(-measurement)
        elif (i == 1):
            measurements.append(measurement + correction_left)
        elif(i == 2):
            measurements.append(measurement - correction_right)

# def generator(samples, batch_size = 128):
#     num_samples = len(samples)
#
#     while 1:
#         shuffle(samples)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
# Normalize and mean center the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((80,20),(0,0))))
model.add(Convolution2D(filters=24, kernel_size=(5,5), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=36, kernel_size=(5,5)(5,5), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=48, kernel_size=(5,5)(3,3), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=64, kernel_size=(5,5)(3,3), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')