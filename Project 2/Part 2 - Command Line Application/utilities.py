import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import math
import argparse
import time
import copy

def get_input_args_train():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', metavar = 'Data Directory (path/to/dir/)', type = str, default = 'flowers/', help = 'Specify path to folder of images to be used for training & validation')
    parser.add_argument('--save_dir', type = str, default = '../checkpoints/', help = 'Specify path to folder to save Model Checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg13', nargs = '?', 
                        choices = ["alexnet", "vgg11_bn", "vgg13", "vgg16", "densenet121","resnet18"],
#                         choices = [ "alexnet", "squeezenet1_0", "vgg13", "vgg16", "densenet121", "googlenet",
#                                    "convnext_tiny", "inception_v3", "shufflenet_v2_x1_0", "mobilenet_v2","resnext50_32x4d", "wide_resnet50_2","mnasnet1_0"
#                                    ], 
                        help = 'Provide model architecture to be be used')
    parser.add_argument('--arch_type', type = str, default = 'existing', nargs = '?', choices = ["existing","new","custom"],
                        help = 'Provide type of model architecture to be be used')
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Provide learning rate to be be used, default is 0.003')
    parser.add_argument('--hidden_units', type = list, default = [512, 256], help = 'Provide number of hidden units to use, default is 128')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Provide number of epochs to use for training')
    parser.add_argument('--gpu', type = str, nargs='?', default = 'cpu', const = 'gpu', help = 'GPU will be used for training if you specific --gpu')
    parser.add_argument('--feature_extract', action='store_false', help = 'Indicate whether to extract features or fine tune the whole existing model')    
    
    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', metavar = 'Path to image (path/to/image)', default = 'flowers/valid/10/image_07094.jpg', type = str, help = 'Specify path to image for which model needs to predict')
    parser.add_argument('--save_dir', type = str, default = '../checkpoints/checkpoint.pth', help = 'Specify path to folder to retrieve Checkpoints')
    parser.add_argument('----category_names', type = str, default = 'cat_to_name.json', help = 'Provide mapping of class indexes to category names')
    parser.add_argument('--top_k', type = int, default = 3, help = 'Provide a number of most likely classes to be returned')    
    parser.add_argument('--gpu', type = str, nargs='?', default = 'cpu', const = 'gpu', help = 'GPU will be used for training if you specific --gpu')
    
    return parser.parse_args()

#Load training & validation data and transform plus load label mapping
def data_transforms():
    return {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
        'valid' : transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    }

def image_datasets(directory, data_transforms):
    return {
        'train' : datasets.ImageFolder(directory + 'train/', transform = data_transforms['train']),
        'valid' : datasets.ImageFolder(directory + 'valid/', transform = data_transforms['valid'])
    }

def data_loaders(image_datasets):
    
    data = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle = True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    }
    #FOR TEST
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(10)))
    data = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], sampler = sampler, batch_size=64),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], sampler = sampler, batch_size=64)
    }
    
    return data

def process_image(image_path):

    im = Image.open(image_path)  
    # Resize
    if im.size[1] < im.size[0]:
        im.thumbnail((255, math.pow(255, 2)))
    else:
        im.thumbnail((math.pow(255, 2), 255))    
        
    #Crop
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    #Convert to np.array
    np_image = np.array(im)/255
    
    #Undo Mean, Standard Deviation and Transpose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])    
    np_image = (np_image - mean)/ std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, is_inception=False):
    
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()                 # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):            # forward  track history if only in train
                    # Get model outputs and calculate loss, In train mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    if is_inception and phase == 'train': # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train': # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history