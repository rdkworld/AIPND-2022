#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import math
from collections import OrderedDict
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# ## Provide user inputs

# In[2]:


#User Inputs in this notebook (or passed as command line arguments in Command Line Application)
in_arg = {}
in_arg['arch'] = 'vgg16'
in_arg['arch_type'] = 'custom' #To differentiate between adding new layers/activations vs just modifying the output features of existing last layer
NUM_CLASSES = 102 #No of flower species
in_arg['feature_extract'] = True #To identify layers prior to classifer or fc should be frozen
in_arg['hidden_units'] = [1024, 512, 256] #Length indicates number of new layers to be added, value at index indicates number of activations to be added each layer
in_arg['learning_rate'] = 0.003
in_arg['epochs'] = 10
in_arg['save_dir'] = 'checkpoints/'


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[3]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[4]:


# TODO: Define your transforms for the training, validation, and testing sets
#data_transforms = 

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
#image_datasets = 
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders
trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)

#Uncomment for testing (small dataset)
# from torch.utils.data.sampler import SubsetRandomSampler
# sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(10)))
# trainloaders = torch.utils.data.DataLoader(train_data, sampler = sampler, batch_size=64)
# validloaders = torch.utils.data.DataLoader(valid_data, sampler = sampler, batch_size=64)
# testloaders = torch.utils.data.DataLoader(test_data, sampler = sampler, batch_size=64)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# <font color='red'>**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.</font>

# In[6]:


#Load network as feature detector
#Get pre-trained model specifications and override with classifier portion with user activation units

#Helper function to freeze the layer if feature extracting
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#Build/design the custom model            
def build_custom_models(model_name, model_type, num_classes, feature_extract, hidden_units, use_pretrained=True):
       
    model_ft = getattr(models, model_name)(pretrained = use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    if model_name == 'resnet18':
        in_features = model_ft.fc.in_features
    else:
        try: #Is there an iterable classifier layer for the model chosen?
            iter(model_ft.classifier)
        except TypeError: #If no, choose the classifier layer with no index
            in_features = model_ft.classifier.in_features
        else:
            try: #If yes, check if first index has in_features attribute
                in_features = model_ft.classifier[0].in_features
            except AttributeError: #If No, check if second index has in_features attribute
                in_features = model_ft.classifier[1].in_features
        
    hidden_layers = [in_features] + hidden_units
    layer_builder = (
        lambda i, v : (f"fc{i}", nn.Linear(hidden_layers[i-1], v)),
        lambda i, v: (f"relu{i}", nn.ReLU()),
        lambda i, v: (f"drop{i}", nn.Dropout())        
    )
    
    layers = [f(i, v) for i, v in enumerate(hidden_layers) if i > 0 for f in layer_builder]
    layers += [('fc_final', nn.Linear(hidden_layers[-1], num_classes)),
               ('output', nn.LogSoftmax(dim=1))]    

    if model_name == 'resnet18':
        fc = nn.Sequential(OrderedDict(layers))
        model_ft.fc = fc
    else:
        classifier = nn.Sequential(OrderedDict(layers))
        model_ft.classifier = classifier
#     print("AFTER")
#     print(model.classifier)
    
    return model_ft


# In[7]:


#Instantiate the model and move to GPU/CPU
model_ft = build_custom_models(in_arg['arch'], in_arg['arch_type'], NUM_CLASSES, in_arg['feature_extract'], 
                               in_arg['hidden_units'], use_pretrained=True)
print(model_ft)
#GPU or CPU Agnostic handling. Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    in_arg['gpu'] = 'gpu'
else:
    device = in_arg['gpu'] = 'cpu'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
print(f"Using {device} device for training/validating")


# In[8]:


#Create Optimizer but first gather parameters to update
params_to_update = model_ft.parameters()
print("Params to learn:")
if in_arg['feature_extract']:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=in_arg['learning_rate']) #

#Set Loss criterion
criterion = nn.NLLLoss() #nn.CrossEntropyLoss() #Potential Hyperparameter


# In[9]:


# TODO: Build and train your network
epochs = in_arg['epochs']
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloaders:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model_ft(inputs) #model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model_ft.eval()
            with torch.no_grad():
                for inputs, labels in testloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model_ft.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloaders):.3f}")
            running_loss = 0
            model_ft.train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[10]:


#First, define the function to validate/test on testing set
def validate(model, criterion, testing_set):
    accuracy = 0
    test_loss = 0
    model.eval() # Evaluation mode
    with torch.no_grad():
        for images, labels in testing_set:
            
            images = images.to(device)
            labels = labels.to(device)

            output = model(images) #model.forward(images)
            test_loss += criterion(output, labels).item()

            # Take exponential to get log softmax probibilities
            probs = torch.exp(output)

            # highest probability is the predicted class
            # compare with true label
            correct_predictions = (labels.data == probs.max(1)[1])

            # Turn ByteTensor into np_array to calculate mean
            accuracy += np.array(correct_predictions).mean()
    
#     model.train() # Switch training mode back on
    
    return test_loss/len(testing_set), accuracy/len(testing_set)


# In[11]:


#Then, call the function and test
# TODO: Do validation on the test set
test_loss, accuracy = validate(model_ft, criterion, testloaders)
print("Network preformance on test dataset-------------")
print("Accuracy on test dataset: {:.2f}%".format(accuracy*100))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[12]:


print("Our model: \n\n", model_ft, '\n')
print("The state dict keys: \n\n", model_ft.state_dict().keys())


# In[13]:


# TODO: Save the checkpoint
checkpoint = {'arch': in_arg['arch'],
              'arch_type': in_arg['arch_type'],
              'class_to_idx' : train_data.class_to_idx,
              'state_dict': model_ft.state_dict(),
              'hidden_units': in_arg['hidden_units'],
              'num_classes' : NUM_CLASSES,
              'feature_extract' : in_arg['feature_extract'],
              'gpu_or_cpu' : in_arg['gpu']
             }
torch.save(checkpoint, 'checkpoint.pth')
print(f"Saved checkpoint as checkpoint.pth")


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[14]:


#GPU/CPU Agnostic Code
if torch.cuda.is_available():
    device = torch.device("cuda")
    checkpoint = torch.load('checkpoint.pth')
else:
    device = "cpu"
    checkpoint = torch.load('checkpoint.pth', map_location = device)
print(f"Using {device} device for predicting/inference")

if checkpoint['arch_type'] == 'custom':
    model_ft = build_custom_models(checkpoint['arch'], checkpoint['arch_type'], len(checkpoint['class_to_idx']), checkpoint['feature_extract'], 
                                   checkpoint['hidden_units'], use_pretrained=True)
else:
    print("Nothing to predict")
    exit()
     
model_ft.class_to_idx = checkpoint['class_to_idx']
model_ft.gpu_or_cpu = checkpoint['gpu_or_cpu']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.to(device)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# ## Sample Display of Original Flower (Neither Actual Label or a Prediction)

# In[15]:


#Choose and display flower as-is before any transformation
image = 'flowers/test/1/image_06743.jpg' #next(iter(testloaders))
display(Image.open(image))


# In[16]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
   
    im = Image.open(image_path)
    
    # Resize the image
    if im.size[1] < im.size[0]:
        im.thumbnail((255, math.pow(255, 2)))
    else:
        im.thumbnail((math.pow(255, 2), 255))    
    #im_resized = im.resize((256, 256))
        
    #Crop image
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    #turn into np.array and standardize
    np_image = np.array(im)/255
    
    #undo mean, std and then transpose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])    
    np_image = (np_image - mean)/ std
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[17]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0)) #image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax


# ## Sample Display of Transformed Flower (Neither Actual Label or a Prediction)

# In[18]:


#Test the function above after Transformation
imshow(process_image(image), title = 'Testing display of flower using imshow() function after Transformation')


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[19]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    tensor_img = torch.FloatTensor([process_image(image_path)])
    #print(tensor_img.unsqueeze(0).shape)
    tensor_img = tensor_img.to(device)
    log_ps = model(tensor_img) #unsqueeze(0)
    #print(log_ps.shape)
    result = log_ps.topk(topk)
    ps = torch.exp(result[0].data).cpu().numpy()[0] #gpu
    #print(ps.shape)
    #print("Between")
    #ps = torch.exp(result[0].data).numpy()[0] #cpu
    #print(ps.shape)
    
    idxs = result[1].data.cpu().numpy()[0]
    #idxs = result[1].data.numpy()[0]
    return (ps, idxs)


# In[20]:


# Get the probabilties and indices from passing the image through the model
probs, idxs = predict(image, model_ft)
print(f"Probabilities are {probs}")
print(f"Indexes are {idxs} which need to be converted to classes which need to converted to class or species name")


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[21]:


get_ipython().system('ls flowers/test/*')


# ## Actual Label (Species of Flower) - PROVIDE YOUR INPUT FLOWER NAME/PATH 

# In[22]:


test_image_path = test_dir + '/12/image_03994.jpg' #UPDATE THIS FOR PREDICTION
test_image_name = cat_to_name[os.path.basename(os.path.dirname(test_image_path))]
                              
# Show the image for reference
imshow(process_image(test_image_path), title=test_image_name)


# ## Predicted Label (Species of Flower)

# In[23]:


# TODO: Display an image along with the top 5 classes
# Get the probabilties and indices from passing the image through the model
probs, idxs = predict(test_image_path, model_ft)

# Swap the keys and values in class_to_idx so that
# indices can be mapped to original classes in dataset
idx_to_class = {v: k for k, v in model_ft.class_to_idx.items()}

# Map the classes to flower category lables                              
names = list(map(lambda x: cat_to_name.get(f"{idx_to_class[x]}",'Unknown'), idxs))

# Display top 5 most probable flower categories                               
y_pos = np.arange(len(names))
plt.barh(y_pos, probs)
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
 
plt.show()


# <font color='red'>**Reminder for Workspace users:** If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
#     
# We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.</font>

# In[24]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace

