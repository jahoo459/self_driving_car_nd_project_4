# **Behavioral Cloning** 

## Project summary

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ../doc/architecture.PNG "Model Visualization"
[image2]: ../doc/line_center.jpg "Line center"
[image3]: ../doc/recovery.jpg "Recovery Image"
[image4]: ../doc/recovery2.jpg "Recovery Image"
[image5]: ../doc/recovery_difficult.jpg "Recovery Image"
[image6]: ../doc/2nd_track.jpg "Normal Image"
[image7]: ../doc/ "Flipped Image"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center 
lane driving, recovering from the left and right sides of the road and some additional records of the most difficult
curves.

At the beginning I recorded only the traces with the car in the middle of the line. The model trained
on such data was very stable and driving smoothly but it could not recover from difficult situations like sharp
curves. Because of that I recorded some additional traces with recovering scenarios. They helped the model
to get back to the middle of the road but also decreased the smoothness of the drive. Now the car is more likely to
jitter a bit. 

Especially difficult scenario was a sharp curve just after the bridge, without red-white marking on the side. I had to record 
multiple additional traces in this place.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a simple architecture containing of 2 convolutional layers and one dense layer. With this net I proved
that everything is working as expected. For the next step I used the proposed NVIDIA net. This model is very powerful and drove the
car well.

During the design process I noticed how significant role the quality of data plays in the process. In my feeling
correct and high quality training data was more important than perfectly tuned model architecture.

During the development I noticed that there are some especially difficult places on the track where the car was leaving the road.
To avoid this I recorded additional data for those scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-90) consisted of a convolution neural network with the following layers and layer sizes:
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track using center lane driving. I did it both clock- and counterclockwise.
Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center.
I took some more recordings of the most difficult curve just after the bridge.

![alt text][image3]
![alt text][image4]
![alt text][image5]

I also additionaly recorded one lap on second track.
![alt text][image6]


To augment the dataset, I also flipped the images from the front camera and assigned -1*steering_angle to them.
It should simulate the behaviour of the car going in the different direction.


After the collection process, I had 14086 * 4 (center, inverse, left, right) training samples and 6038 *4 validation samples. 

I then preprocessed this data by cropping the camera view and normalizing the images. This is done in train.py at lines 76-77.

For the training I used the Adam optimizer so there was no need to manually tune the learning rate.
To find the best number of epochs I used a EarlyStopping callback that stops the training when the validation loss does
not decrease anymore.
```sh
EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
```

The final training loss was 0.0207 and validation loss 0.0215. This is a good result showing that the net 
is not overfitting (on the other hand it has to be considered that all the data are quite similar as they
were taken on the same track).

The video from a final lap can be found here: https://www.youtube.com/watch?v=eW0_nDboJe8
