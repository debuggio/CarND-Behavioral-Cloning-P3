**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[InitialImages]: ./images/InitialImages.png "Initial images"
[ProcessedImages]: ./images/ProcessedImages.png "Processed images"
[Model]: ./images/Model.png "Model"
[OriginalImage]: ./images/original.jpg "Initial image"
[FlippedImage]: ./images/flipped.jpg "Flipped image"
[DriveCenter]: ./images/drive_center.jpg "Center"
[Optimal]: ./images/Optimal.png "Optimal"

## Rubric Points

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing automated run

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it should be self-explanatory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA model, which was mentioned in one of the videos about the project.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

**Model layers:**
* Normalization, to avoid saturation
* Convolution: 24, 5x5 activation: 'relu'
* Convolution: 36, 5x5 activation: 'relu'
* Convolution: 48, 5x5 activation: 'relu'
* Convolution: 64, 3x3 activation: 'relu'
* Convolution: 64, 3x3 activation: 'relu'
* Dropout: 50%
* Fullyconnected: 100
* Fullyconnected: 50
* Fullyconnected: 10
* Fullyconnected: 1

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. This approach gave me around 8% improvements in my previous Lab, so I used the same layer with tha same probability here

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. Also I tried to not reduce speed on sharp turns, so sometimes I was driving closer to one of the sides and then returned to the center of the road, so I'm expecting that model has enough information about what to do if it almost out of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a simple convolution neural network model similar to the one that was shown in video. This was a good starting point, to check that car can drive in general, unfortunately, right off the road.
THen I added an image preprocessing step to crop image and remove sky and trees as well as car hood. After that I added images from left and right camera with correction. I found my model underfitted, because training and validation errors were pretty high and car wasn't able to make 3 turns. To double my input data and to add right turns I added an image preprocessing step that flips images horizontally.

To combat the overfitting, I added dropout step to model with 50% keep probability, because this value worked in my previous project

After that car was able to drive around 30% of the thrack

Last improvement that gave a huge impact was to record a few situations when car is going to leave the track, but then turns back to the middle of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
**Model layers:**
* Normalization, to avoid saturation
* Convolution: 24, 5x5 activation: 'relu'
* Convolution: 36, 5x5 activation: 'relu'
* Convolution: 48, 5x5 activation: 'relu'
* Convolution: 64, 3x3 activation: 'relu'
* Convolution: 64, 3x3 activation: 'relu'
* Dropout: 50%
* Fullyconnected: 100
* Fullyconnected: 50
* Fullyconnected: 10
* Fullyconnected: 1

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model][Model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center][DriveCenter]

The second lap I tried to not reduce speed and in try to use optimal trajectory in sharp turns  :

![Optimal][Optimal]

To augment the data sat, I also flipped images and angles thinking that this would double my training data, plus it would add right turns. For example, here is an image that has then been flipped:

![Original image][OriginalImage]
![Flipped image][FlippedImage]


After the collection process, I had 1909 number of data points. I then preprocessed this data by flipping images and using pictures from all 3 available cameras (that gave me in total 5727 images). I decided not to use additional preprocessing, like ajust brightness, because it's not about computer vision :)

**Initial steering distribution**

![initial distribution][InitialImages]

**Steering istribution after processing**

![Processed images][ProcessedImages]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
