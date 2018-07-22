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

from sklearn.model_selection import train_test_split


#train_samples, validation_samples = train_test_split(samples, test_size=0.2)
i = 0;
size = np.array(lines).shape[0]
print(size)
for line in lines:

    #source_path = line[i]
    #filename = source_path.split('/')[-1]
    current_path = './data_new'
    #image = cv2.imread(current_path)
    
    # read in images from center, left and right cameras
    #print(current_path + line[0])
    img_center = cv2.imread(current_path + line[0])
    #print(np.array(img_center).shape)
    img_left = cv2.imread(current_path + line[1])
    img_right = cv2.imread(current_path + line[2])
    steering_center = float(line[3])
    #if i >= size-2000:
    #    print(i)
    #    print(np.array(img_center).shape)    
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
    #angles.extend(steering_center, steering_left, steering_right)
        #images.append(image)
        #angle = float(line[3])
        #angles.append(angle)
    #print(np.array(images).shape)
    i = i + 1
print('images shape')
print(np.array(images).shape)
"""
lines2 = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines2.append(line)
for line in lines2:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        angle = float(line[3])
        angles.append(angle)
print(np.array(images).shape)  

"""
print('augmented images shape')
augmented_images, augmented_angles = [], []

for image, angle in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_angles.append(angle*-1.0)
    
print(np.array(augmented_images).shape)        
X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)       
        
        
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                    
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)        
     
# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)        
        
        
"""       
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


#ch, row, col = 3, 80, 320  # Trimmed image format


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
#model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(ch, row, col),output_shape=(ch, row, col)))
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
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3)
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1)
"""
model.save('model2.h5')