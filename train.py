import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split

samples = []
data = "./data/driving_log.csv"

with open(data) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.3)

images = []
measurements = []
correction_right = -0.2
correction_left = 0.2
batch_size = 32

# for line in samples:
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('\\')[-1]
#         current_path = './data/IMG/' + filename
#
#         image = cv2.imread(current_path)
#         images.append(image)
#         if(i == 0):
#             images.append(np.fliplr(image))
#
#         measurement = float(line[3])
#         if(i == 0):
#             measurements.append(measurement)
#             measurements.append(-measurement)
#         elif (i == 1):
#             measurements.append(measurement + correction_left)
#         elif(i == 2):
#             measurements.append(measurement + correction_right)

def generator(samples, batch_size = 32):
    num_samples = len(samples)

    while 1: #loop forever
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                centered_img_name = './data/IMG/' + batch_sample[0].split('\\')[-1]
                left_img_name = './data/IMG/' + batch_sample[1].split('\\')[-1]
                right_img_name = './data/IMG/' + batch_sample[2].split('\\')[-1]
                centered_img = cv2.imread(centered_img_name)
                left_img = cv2.imread(left_img_name)
                right_img = cv2.imread(right_img_name)
                steering_angle = float(batch_sample[3])

                images.append(centered_img)
                images.append(np.fliplr(centered_img))
                images.append(left_img)
                images.append(right_img)
                measurements.append(steering_angle)
                measurements.append(-1 * steering_angle)
                measurements.append(steering_angle + correction_left)
                measurements.append(steering_angle + correction_right)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


ch, row, col = 3, 160, 320

train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
# Normalize and mean center the images
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((80,20),(0,0))))
model.add(Convolution2D(filters=24, kernel_size=(5,5), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=36, kernel_size=(5,5), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=48, kernel_size=(3,3), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu', bias_initializer='RandomNormal', strides=(2,2)))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
# model.load_weights('model.h5')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

# model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=10, verbose=1, callbacks=[early_stop])
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=15, verbose=1, callbacks=[early_stop])

model.save('model.h5')