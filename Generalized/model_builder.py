"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
from torchvision import models
from torch import nn
from collections import OrderedDict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def update_last_layer_pretrained_model(pretrained_model, num_classes, feature_extract):
    set_parameter_requires_grad(pretrained_model, feature_extract)
    if hasattr(pretrained_model, 'fc') and 'resnet' in pretrained_model.__class__.__name__.lower(): #resnet
        num_ftrs = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(num_ftrs, num_classes, bias = True)
    elif hasattr(pretrained_model, 'classifier[6]') and ('alexnet' in pretrained_model.__class__.__name__.lower() or 'vgg' in pretrained_model.__class__.__name__.lower()): #alexNet, vgg
        num_ftrs = pretrained_model.classifier[6].in_features
        pretrained_model.classifier[6] = nn.Linear(num_ftrs, num_classes, bias = True)
    elif hasattr(pretrained_model, 'classifier[1]') and 'squeezenet' in pretrained_model.__class__.__name__.lower(): #squeezenet
        pretrained_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        pretrained_model.num_classes = num_classes
    elif hasattr(pretrained_model, 'AuxLogits.fc') and 'inception' in pretrained_model.__class__.__name__.lower(): #inception
        num_ftrs = pretrained_model.AuxLogits.fc.in_features 
        pretrained_model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes) #Auxilary net
        num_ftrs = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(num_ftrs,num_classes) #Primary net
    elif hasattr(pretrained_model, 'classifier') and 'densenet' in pretrained_model.__class__.__name__.lower(): #densenet
        num_ftrs = pretrained_model.classifier.in_features
        pretrained_model.classifier = nn.Linear(num_ftrs, num_classes, bias = True)
    elif hasattr(pretrained_model, 'heads') and 'vit' in pretrained_model.__class__.__name__.lower(): #vit transformer
        num_ftrs = pretrained_model.heads.head.in_features
        pretrained_model.heads.head = nn.Linear(num_ftrs, num_classes, bias = True)
    elif hasattr(pretrained_model, 'head') and 'swin' in pretrained_model.__class__.__name__.lower(): #swin transformer
        num_ftrs = pretrained_model.head.in_features
        pretrained_model.head = nn.Linear(num_ftrs, num_classes, bias = True)

    return pretrained_model

def initialize_existing_models(model_name, model_type, num_classes, feature_extract, hidden_units, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                        nn.Linear(num_ftrs, num_classes),
                        nn.LogSoftmax(dim=1))
        input_size = 224
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
                                    nn.Linear(num_ftrs,num_classes),
                                    nn.LogSoftmax(dim=1))        
        input_size = 224
    elif model_name in ["vgg11_bn", "vgg13", "vgg16"]:
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
                                    nn.Linear(num_ftrs,num_classes),
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Sequential(
                                    nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)),
                                    nn.LogSoftmax(dim=1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet121":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
                                    nn.Linear(num_ftrs, num_classes),
                                    nn.LogSoftmax(dim=1))     
        input_size = 224
    elif model_name == "inception": # This model expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, num_classes),
                                    nn.LogSoftmax(dim=1))
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                            nn.Linear(num_ftrs,num_classes),
                            nn.LogSoftmax(dim=1))
        input_size = 299
    else:
        print("Invalid model name, please use one of the models supported by this application, exiting...")
        exit()
    return model_ft, input_size

#Get pre-trained model specifications and override with classifier portion with user activation units
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

#Define model/ neural network class
# class ImageClassifier(nn.Module):
#     def __init__(self):
#         super(ImageClassifer, self).__init__()
#         self.flatten = nn.Flatten()
#         self.model_stack = nn.Sequential(
#             nn.Linear(),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(),
#             nn.LogSoftmax(dim=1)
#         )
               
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.model_stack(x)
#         return logits

# class TinyVGG(nn.Module):
#     """Creates the TinyVGG architecture.

#     Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
#     See the original architecture here: https://poloclub.github.io/cnn-explainer/

#     Args:
#     input_shape: An integer indicating number of input channels.
#     hidden_units: An integer indicating number of hidden units between layers.
#     output_shape: An integer indicating number of output units.
#     """
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
#         super().__init__()
#         self.conv_block_1 = nn.Sequential(
#           nn.Conv2d(in_channels=input_shape, 
#                     out_channels=hidden_units, 
#                     kernel_size=3, 
#                     stride=1, 
#                     padding=0),  
#           nn.ReLU(),
#           nn.Conv2d(in_channels=hidden_units, 
#                     out_channels=hidden_units,
#                     kernel_size=3,
#                     stride=1,
#                     padding=0),
#           nn.ReLU(),
#           nn.MaxPool2d(kernel_size=2,
#                         stride=2)
#         )
#         self.conv_block_2 = nn.Sequential(
#           nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
#           nn.ReLU(),
#           nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
#           nn.ReLU(),
#           nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#           nn.Flatten(),
#           # Where did this in_features shape come from? 
#           # It's because each layer of our network compresses and changes the shape of our inputs data.
#           nn.Linear(in_features=hidden_units*13*13,
#                     out_features=output_shape)
#         )
    
#     def forward(self, x: torch.Tensor):
#         x = self.conv_block_1(x)
#         x = self.conv_block_2(x)
#         x = self.classifier(x)
#         return x
#         # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion
