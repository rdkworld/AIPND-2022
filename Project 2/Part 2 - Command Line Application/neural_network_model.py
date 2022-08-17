from torchvision import models
from torch import nn
import torch
from collections import OrderedDict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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
