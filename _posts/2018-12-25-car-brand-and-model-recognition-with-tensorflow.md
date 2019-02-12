---
layout: post
title: "Car Brand and Model Recognition with Tensorflow"
categories: deep-learning
---

Image Classification is the problem of labeling Images based on the contextual data of the image.
Car detection and classification is an important task in many fields such as traffic management and control, transportation, etc.
We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.


#### Data exploration

The dataset structure comes in this way
```shell
|-ann_test.csv
|-ann_train.csv
|-names.csv
|-Test
|     |- Acura Integra Type R 2001
|                                 |- 00000.jpg
|                                 |- 00001.jpg
|                                 |- ....
|     |- ......
|-Train
|     |- Acura Integra Type R 2001
|                                 |- 00000.jpg
|                                 |- 00001.jpg
|                                 |- ....
|     |- ......
```

#### Data preprocessing

__A.Boundary boxes:__

  The first preprocessing method that should be applied on the data set, is cropping cars boundary boxes provided in `annotation_train.csv` file.

__B. Data augmentation:__
  Data augmentation is the process of increasing the data set size by generating new data derived from the original training data.
This technique Shows a large enhancement to the accuracy in many computer vision and deep learning tasks. But generating useless and duplicated data could lead to overfitting and/or accuracy dropping.
**Some recommended techniques for data augmentation are:**
- Flipping with Fixed degrees(90, 180, 270, etc)
- Translating
- Mixed techniques (Flipping and translating)
- Noise addition

**And some others that are not recommended:**

- Changing images rgb values.
- Flipping with random degrees.


#### Approach and Algorithms:

##### __Transfer learning:__

  In our project, we implement, train, and test state-of-the-art classifiers trained on domain general datasets for the task of identifying the make and models of cars from various angles and different settings, with the added constraint of limited data and time. We experiment with different levels of transfer learning for fitting these models over to our domain.
We chose mobilenet_v1_0.50  model for training.

###### __Mobilenet (224 ,0.5):__

  MobileNets are small, low-latency, low-power CNN models parameterized to meet the resource constraints of a variety of use cases. 
__"Convolutional"__ just means that the same calculations are performed at each location in the image.
We chose it to be compatible with the resources on hands (On both hardware and time resources).
The MobileNet is configurable in two ways:
Input image resolution: 128,160,192, or 224px.
 Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy.
The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25.


![Mobilenet](../../../../images/mobilenet.png?raw=true "Mobilenet")

> mobile net pretrained model compared to other models The latency and power usage of the network scales with the number of Multiply-Accumulates (MACs) which measures the number of fused Multiplication and Addition operations. We used the most suitable one for the resources on the hands.

## Implementation

### Data processing

__Cropping images__

  ![cropping](../../../../images/cropping.png?raw=true "Cropping")

### Retraining (Transfer Learning)

> The machine used here is macbook 2017 with `2.3 GHz Intel Core i5 8th gen` and `8 GB Ram`

We used [official tensorflow retraining image classifiers](https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py) scripts to retrain the MobileNet model on our dataset.

## __Steps__

### **1) Bottlenecks**:
> is an informal term we often use for the layer just before the final output layer that actually does the classification. 

The first phase analyzes all the images on disk and calculates and caches the bottleneck values for each of them. TensorFlow calls this an "image feature vector"

The reason final layer retraining can work on new classes (196 car models) is that it turns out the kind of information needed to distinguish between all the 1,000 classes in ImageNet is often also useful to distinguish between the new kinds of cars.

Because every image is reused multiple times during training and calculating each bottleneck takes a significant amount of time, it speeds things up to cache these bottleneck values on disk so they don't have to be repeatedly recalculated.


### **2) Training**:

Once the bottlenecks are complete, the actual training of the top layer(classification layer) of the network begins.
The logging prints 3 differents info every 10 epochs:

- __The training__ accuracy shows what percent of the images used in the current training batch were labeled with the correct class.

- __The validation__ accuracy is the precision on a randomly-selected group of images from a different set.

- __Cross entropy__ is a loss function which gives a glimpse into how well the learning process is progressing.


After retraining for 1000, 10000, 50000 and 100000 epochs we found out that 50000 epochs is the most suitable one and that the train and validation accuracy are converged after that number.

For each training step(epoch) the script:

- chooses ten images at random from the training set.
- finds their bottlenecks from the cache.
- feeds them into the final layer to get predictions.

Those predictions are then compared against the actual labels to update the final layer's weights through the back-propagation process

Training with 50000 on the training set (8,144 Image) takes around 3.5 hours.

### **2) Testing and Generating Submission file**:

The testing file works as:

- Read the testing file parameters throw the cli we need to define the data file, model file, labels file and annotation file using `argparse` method. 

- Load the graph model and the graph diff that resulted out from the training using a method called `load_graph`

- Open a new tensorflow session

- Iterate over all testing images and for each image feed it to the trained CNN, find the id of the returned label and write it to the file.

Reading 1 image and feed it to the network and write it back to the result file takes around 1.9 seconds so the total testing takes around 4.25 hours


The total accuracy calculated by using mobile net and only cropping in preprocessing is __0.45356__


### Part 2 (aka. Future work)

__0.45356__ is a low accuracy but comparing it to the model size and the preprocessing done (cropping only),data set size  and hardware resources it's a break through and could be highly improved

also this number is still higher than all three methods applied in Caffenet to solve this task according to this [Paper](http://cs231n.stanford.edu/reports/2015/pdfs/lediurfinal.pdf)  by Stanford AI team

```
CaffeNet fine-tuned: 0.447 
CaffeNet partial-train 0.418
CaffeNet full-train 0.417
CaffeNet scratch 0.005 
```

So to increase this number we can work on both the data processing and the retraining

In the first iteration of solving this task we didn't apply any preprocessing to the images. And the dataset is relatively small.

- Applying Data agumantation methods discussed before to increase the training set.

- Applying Grayscale to the images

- Use Larger models like GoogleNet

Applying those enhancement will make a huge affect on the total test accuracy 
