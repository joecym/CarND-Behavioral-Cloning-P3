# **Behavioral Cloning**
# Joe Cymerman
# 22 July 2018
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I also changed the speed in drive.py from 9mph to 15mph and successfully navigated the track. Any faster, however, I saw oscillatory behavior in the car's driving. 

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 62-75). This was the same network shown in the video by NVIDIA.  

Here is the architecture:

``` python
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
```
#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer after the convolutional layers in order to reduce overfitting (model.py line 70). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 78). I used a split of 15% for the validation data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I trained entirely on the first track, but drove several laps in both directions. I also collected more images around the corners that were giving the car trouble.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adjust and tune the NVIDIA model outlined in the previous video. At first, I combined my data with the test data set provided by Udacity, but I found my model performing worse. I seemed to do better with less data that was entirely my own.

I augmented my data by flipping the images, performed a normalization, and also adjusted the steering angle for the left and right camera images by 0.15.

I used a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because it works! I added a dropout layer after the convolutional layers, but otherwise, it is the same.

It still seemed like my model was overfitting, because my validation error increased. Here is the output of my command window:

``` sh
Train on 31308 samples, validate on 5526 samples
Epoch 1/10
2018-07-22 21:21:43.237910: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-07-22 21:21:43.237981: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-07-22 21:21:43.237993: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-07-22 21:21:43.238000: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-07-22 21:21:43.238007: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2018-07-22 21:21:43.325887: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-07-22 21:21:43.326265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2018-07-22 21:21:43.326313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
2018-07-22 21:21:43.326338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
2018-07-22 21:21:43.326357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0)-> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
31308/31308 [==============================] - 35s - loss: 0.0383 - val_loss: 0.0521
Epoch 2/10
31308/31308 [==============================] - 33s - loss: 0.0353 - val_loss: 0.0536
Epoch 3/10
31308/31308 [==============================] - 33s - loss: 0.0340 - val_loss: 0.0501
Epoch 4/10
31308/31308 [==============================] - 33s - loss: 0.0324 - val_loss: 0.0491
Epoch 5/10
31308/31308 [==============================] - 33s - loss: 0.0309 - val_loss: 0.0531
Epoch 6/10
31308/31308 [==============================] - 33s - loss: 0.0297 - val_loss: 0.0535
Epoch 7/10
31308/31308 [==============================] - 33s - loss: 0.0280 - val_loss: 0.0562
Epoch 8/10
31308/31308 [==============================] - 33s - loss: 0.0266 - val_loss: 0.0535
Epoch 9/10
31308/31308 [==============================] - 33s - loss: 0.0251 - val_loss: 0.0587
Epoch 10/10
31308/31308 [==============================] - 33s - loss: 0.0232 - val_loss: 0.0626
```

To combat the overfitting, I tried fewer epochs, as low as three. However, I found that the car performed better with more epochs, even though my validation loss kept increasing. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially the second curve after the bridge. To improve the driving behavior in these cases, I took more data around these curves and adjusted the steering adjustment. I also tried adjusting the layers in the network, but I didn't see any improvement there. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I was even able to incease the speed to 15mph. I tried up to 30mph, because this is the speed I was going when training. However, the steering would oscillate at speeds higher than 15 mph and move off the track.

#### 2. Final Model Architecture

The final model architecture is the same as listed above.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track, and tried to just stay on the track as best I could. I found the keyboard controls a bit challenging at first.

Then I repeated this process on the same track in the opposite direction.

To augment the data sat, I also flipped images and angles thinking that this would provide a more balanced learning set.

After the collection process, I had 18,417 number of data points. After processing the data, and segregating the data set, my training set had 31,308 samples.


I finally randomly shuffled the data set and put 15% of the data into a validation set. 
