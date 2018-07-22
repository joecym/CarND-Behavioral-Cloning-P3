import os
import csv
import cv2
import numpy as np
import sklearn

lines = []
images = []
angles = []

with open('data_new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    current_path = './data_new'
    
    # read in images from center, left and right cameras
    img_center = cv2.imread(current_path + line[0])
    img_left = cv2.imread(current_path + line[1])
    img_right = cv2.imread(current_path + line[2])
    steering_center = float(line[3])
   
    # create adjusted steering measurements for the side camera images
    correction = 0.15 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # add images and angles to data set
    #images.extend(img_center, img_left, img_right)
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    angles.append(steering_center)
    angles.append(steering_left)
    angles.append(steering_right)

# Here I'm checking the shape of the input array for troubleshooting
print('images shape')
print(np.array(images).shape)
print('augmented images shape')
augmented_images, augmented_angles = [], []

# flip the images to double the data set size
for image, angle in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_angles.append(angle*-1.0)

#checking the final input size    
print(np.array(augmented_images).shape) 

X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)       
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.15,shuffle=True,nb_epoch=10)

#save the model
model.save('model2.h5')