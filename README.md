# **Traffic Sign Recognition** 


### This is udacity Self-Driving Car NanoDegree Project. All the realted file/picture could find at my [GitHub Page](https://github.com/ChengZhongShen/Traffic_Sign_Classifier.git)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples_2/read_me_pictures/sample_fig.jpg "43 sign"
[image1a]: ./examples_2/read_me_pictures/samples_with_label.png "sample with label"
[image2]: ./examples_2/read_me_pictures/sample_distrution.png "sample distrubution"
[image3]: ./examples_2/read_me_pictures/rgb_2_gray.png "Gray Scale"
[image3a]: ./examples_2/read_me_pictures/data_gen.png "data gen"
[image4]: ./examples/1_Speed_limit_30.jpg "Traffic Sign 1"
[image5]: ./examples/4_speed_limit_70.jpg "Traffic Sign 4"
[image6]: ./examples/12_priority_road.jpg "Traffic Sign 12"
[image7]: ./examples/15_no_vehicles.jpg "Traffic Sign 15"
[image8]: ./examples/17_no_entry.jpg "Traffic Sign 17"
[image9]: ./examples/20_Dangersous_curve_to_right.jpg "Traffic Sign 20"
[image10]: ./examples/23_slippery_road.jpg "Traffic Sign 23"
[image11]: ./examples/25_Road_work.jpg "Traffic Sign 25"
[image12]: ./examples/28_Children_crossing.jpg "Traffic Sign 28"
[image13]: ./examples/30_beware_of_snow.jpg "Traffic Sign 30"
[image14]: ./examples/35_ahead_only_2.jpg "Traffic Sign 35"
[image15]: ./examples/40_roundabout_mandatory.jpg "Traffic Sign 40"
[image16]: ./examples_2/read_me_pictures/web_image_resized.png "resized to 32*32"
[image17]: ./examples_2/read_me_pictures/web_image_prob.png "image_prob"
[image18]: ./examples_2/read_me_pictures/prob_wrong_image.png "image_prob"
[image19]: ./examples_2/read_me_pictures/feature_map_2.png "image_prob"
[image20]: ./examples_2/read_me_pictures/feature_map_3.png "image_prob"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data Set Summary

The data set is [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32
* The number of unique classes/labels in the data set is 43



#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. 
Below is a picture of 43 kind of trafic sign

![alt text][image1]

Below is a picture of 12 trafic sign with it Label number and label content.

![traffic sign][image1a]

Below is a bar chart showing how the data distrubiion

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Data preprocesse

As a first step, I decided to convert the images to grayscale because according the paper[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that one color image work better than color one.

This is combined with Model use tensoflow fucntion:
```python
# Change RGB to Gray
x = tf.image.rgb_to_grayscale(x) 
```
Here is an example of a traffic sign image of color and gray.

![alt text][image3]

After color change, the data is normalized, because the normalized data could benifit the model training, this is also done by the tensorflow function and combined into the modul:
```python
# normalize the data
x = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)
```

With the origal data, use LeNet_4x model could achieve 94.3% accuracy
Generate more fake data, use LeNet_4x modle could achieve 96.7% accuracy


Four kind of method to generate data, rotate, translation, scale.
Use the OpenCV to handle the images.

Rotate, Rotate the image randomly (-15, 15) degree.
the Code as below:
```python
angle = np.random.randint(-15, 15)
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated_images[i] = cv2.warpAffine(images_sample[i], M, (32, 32)
```

Translation, move the image 6 pix at x, y direction randomly.
```python
offset_x = np.random.randint(-6, 7)
offset_y = np.random.randint(-6, 7)
M = np.float32([[1,0,offset_x], [0,1,offset_y]])
translation_images[i] = cv2.warpAffine(images_sample[i], M, (32, 32))
```

Scale the image to 90%, resize the image from (32, 32) to (28, 28), then pad the image to (32, 32)
```python
resized_image_90 = cv2.resize(images_sample[i], (28, 28))
resized_images_90[i] = cv2.copyMakeBorder(resized_image_90, 2,2,2,2, cv2.BORDER_REPLICATE)
```

Scale the image to 110%, resize the image from (32, 32) to (36, 36), then crop the image to (32, 32).
```python
resized_image_110 = cv2.resize(images_sample[i], (36, 36))
resized_images_110[i] = resized_image_110[2:34,2:34,:]
```

Here is an example of an original images and the generated images:
the first row is oringal image;
second row is rotated image;
third row is tranlated iamge;
fourth row is 90% image;
fifth row is 110% image.

![alt text][image3a]
 

#### 2 Model Architechture

The final model is a expanded LeNet net work, the Conv layer1 and Conv layer 2 are expand 4x.
Detail as following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| RGB to Gray           | 32X32X3 RGB to 32X32X1 Gray                   | 
| Normlize the image    | use tensonflow.image function                 |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Dropout               |                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU                  |                                               |
| Dropout               |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x64                    |
| Fully connected		| Input 1600, output 480						|
| Fully connected       | Input 480, output 336                         |
| Fully connnected      | Input 336, output 43                          |
| Softmax				|            									|


#### 3. Model training

Write a training function to train the model. 
the below factor could be adjust:
* 1. reg_loss, adjust the the regulization loss constant. (0 default)
	reg_loss, 0.01 is OK for the model, this could let the traing process more smooth. if the reg_loss is 0, there will be accuracy joggle.
* 2. Dropout at convolution layer. (1.0 default)
	Dropout at convolution layer ```keep_prob_conv_input```, this is not help in training, keep 1.0.	
* 3. Dropout at the fullyconnect layer. (1.0 default)
	Droput at fully connenct layer ```keep_prob_fc_input``` set at 0.5 will improve the test accucy.(reduce the model overfitting)	
* 4. Training Epoch. (10 default)
	More Epoch will not increase the test accurcy.
* 5. Batch_size. (128 default)
	the Epoch and batch_size is Ok at default.

```python
def train(X_train,y_train,X_valid,y_valid,X_test,y_test, 
          reg_constant_input=0, keep_prob_conv_input=1.0, keep_prob_fc_input=1.0, epoches=10, batch_size=128): 
```

Data quatity, expand the data 5x will increase the test accuccy about 2%.


#### 4 final result

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.978
* test set accuracy of 0.967

 

### Test a Model on New Images

#### 1. Choice test images

Here are 12 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15]

The 10 image might be difficult to classify because the feature (the snow sybol in the middle will be difficult to recognize even by hunman eyes after the image resize to 32*32)

below is the images resize to 32*32 which will feed to the trained neronetwork.
![resized image][image16]


#### 2. The test Result

Here are the results of the prediction:

| Image 		                |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| 12 Priority road	        	| 12 Priority road								| 
| 15 No Vehicles            	| 15 No Vehicles								|
| 17 No Entry		        	| 17 No Entry   								|
| 1 Speed Limit 30	        	| 1 Speed Limit 30					 			|
| 20 Dangerous curver to right  | 20 Dangerous curver to right                  |
| 23 Slippery Road	    		| 23 Slippery Road      						|
| 25 Road Work                  | 25 Road Work                                  |
| 28 Children crossing          | 28 Children crossing                          |
| 30 Beware of ice/snow         | *23 Slippery Road*                            |
| 35 Ahead Only                 | 35 Ahead Only                                 |
| 40 Roundabout Mandatory       | 40 Roundabout Mandatory                       |
| 7 Speed Limit 70              | 7 Speed Limit 70                              |


The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of 91.7%.  It maybe will be better if crop the image include more sign area.

#### 3. The Softmax problity

For most of the image, the model is relatively make a good prodiction, from the bar chart in below, 11 of them have the problity > 0.8, except the 9th image.(first row is the 5 highest probilty, the second row is the test images.)


![problity][image17]

Below is the 5 probility of the 'Beware of ice/snow', it true label is 30, but the pridiction is 23 ('Slippery Road')

![problity_wrong][image18]

###  Visualizing the Neural Network 

Choice the two image make the feature map.
From the feature map, the edage and shape was catch in the first convoluation layer.

![feature_map_2][image19]

![feature_map_3][image20]
