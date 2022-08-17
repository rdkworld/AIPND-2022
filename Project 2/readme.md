# Project 2 - Developing an Image Classifier with Deep Learning

## Overview

In this project, we will train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice will will train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='Part 1 - Develop Image Classifier on Jupyter Notebook/assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

There are two parts to this project.

**Part 1**

In first part of the project, Jupyter notebook was used to work through and implement an image classifier with PyTorch. Jupyter Notebook in .pynb format and also as an HTML file (.html) have been added to *Part 1 - Develop Image Classifier on Jupyter Notebook* folder in this repo.

**Part 2**

In second part of the project, newly built and trained deep neural network on the flower dataset was converted into a command line application so that others can use. This application will be a pair of Python scripts (*train.py and predict.py*) that run from the command line.

## Pre-Trained Models available

Below models/architecture have been trained on the flower dataset and either of these can be used to predict the species for a user provided flower image

* alexnet
* vgg11_bn
* vgg13
* vgg16
* resnet18
* densenet121

Example usage 
    ```python
    --arch resnet18
    ```

Each of the above models can be run two modes:

**1. Existing**

In this mode, existing pretrained models available Pytorch library will be used and only the last layer will be re-trained/customized for our flower dataset to predict one of the 102 classes. This mode can be specified in command line by specifying below option 
```python
--arch_type existing
```

This would be the default if nothing is specified and usually gives better accuracy.

**2. Custom**

In this mode, while existing pretrained models available Pytorch library will still be used, all the layers under either classifier or fully connected section of model architecture will be replaced/customized by user provided number of hidden layers and number of activation units per layer. This mode can be specified in command line by specifying below option 

* To specify mode
    ```python
    --arch_type existing
    ```
* To specify number of hidden layers and number of activations per layer
    ```python
    --hidden_units 1024 512 
    ```
The default hidden_units if arch_type existing is chosen would be [1024, 512] if nothing is specified. This means two additional hidden layers, with first layer having 1024 activation units and second layer having 512 units will be added and by default, ReLU activation and Dropout layer will be added. Finally, at the end, a LogSoftMax non-linear function would be added to make the prediction.

## How to use the Command Line Application

**1. Train**

Train a new network on a data set with *train.py*

* Basic usage: 
  ```python
     python train.py data_directory
  ```

* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
  
  * Set directory to save checkpoints: 
  ```python
     python train.py data_dir --save_dir save_directory
  ```
  * Choose architecture: 
  ```python
    python train.py data_dir --arch "vgg13"
  ```    
  * Set hyperparameters: 
  ```python
     python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
  ```  
  * Use GPU for training: 

  ```python
     python train.py data_dir --gpu 
  ```    

The above mentioned models have been trained with below options

        python train.py flowers/ --arch alexnet --arch_type custom --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch densenet121 --arch_type custom --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch vgg13 --arch_type custom --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch alexnet --arch_type existing --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch densenet121 --arch_type existing --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch vgg13 --arch_type existing --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch resnet18 --arch_type existing --epochs 10 --gpu >> log.txt
        python train.py flowers/ --arch resnet18 --arch_type custom --epochs 10 --gpu >> log.text    

**2. Predict**

Predict flower name from an image with *predict.py* along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: 
  ```python
     python predict.py /path/to/image checkpoint
  ```
* Options:
  * Return top K most likely classes: 
  ```python
     python predict.py input checkpoint --top_k 3
  ```
  * Use a mapping of categories to real names: 
  ```python
  python predict.py input checkpoint --category_names cat_to_name.json
  ```
  * Use GPU for inference: 
  ```python
  python predict.py input checkpoint --gpu   
  ```


## Program Outline

- Time your program
    - Use Time Module to compute program runtime
- Get program Inputs from the user
    - Use command line arguments to get user inputs
- Create Pet Images Labels
    - Use the pet images filenames to create labels
    - Store the pet image labels in a data structure (e.g. dictionary)
- Create Classifier Labels and Compare Labels
    - Use the Classifier function to classify the images and create the classifier labels
    - Compare Classifier Labels to Pet Image Labels
    - Store Pet Labels, Classifier Labels, and their comparison in a complex data structure (e.g. dictionary of lists)
- Classifying Labels as "Dogs" or "Not Dogs"
    - Classify all Labels as "Dogs" or "Not Dogs" using dognames.txt file
    - Store new classifications in the complex data structure (e.g. dictionary of lists)
- Calculate the Results
    - Use Labels and their classifications to determine how well the algorithm worked on classifying images
- Print the Results

Above tasks will be repeated for each of the three image classification algorithms.

## Final Results

| Type  | Count |
| ------------- | ------------- |
| # Total Images  | 40  |
| # Dog Images  | 30  |
| # Not-a-Dog Images  | 10  |

| CNN Model Architecture  | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| **ResNet**  | 90.0%  | 100.0%  | 90.0%  | 82.5%  |
| **AlexNet**  | 100.0%  | 100.0%  | 80.0%  | 75.0%  |
| **VGG**  | 100.0%  | 100.0%  | 93.3%  | 87.5%  |

**Conclusion**

Given above results, the **"best"** model architecture is **VGG**. It outperformed both of the other architectures when considering principal objectives 1 and 2. We noticed that ResNet did classify dog breeds better than AlexNet, but only VGG and AlexNet were able to classify "dogs" and "not-a-dog" at 100% accuracy. The model VGG was the one that was able to classify "dogs" and "not-a-dog" with 100% accuracy and had the best performance regarding breed classification with 93.3% accuracy.

## Source Code

All the source code is available under **Project 2** folder including input sample images.
