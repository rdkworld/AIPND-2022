import torch
from torch import nn
import torch.optim as optim
from collections import OrderedDict

#User Defined
from neural_network_model import initialize_existing_models, build_custom_models, set_parameter_requires_grad
from utilities import get_input_args_train, data_transforms, image_datasets, data_loaders, train_model

#0. Get arguments/hyperparameters from user or set the fefault
in_arg = vars(get_input_args_train())
NUM_CLASSES = 102
print("User arguments/hyperparameters or default used are as below")
print(in_arg)
#Optional input parameters
#DATA_DIR = ''
#MODEL_NAME = ''
#BATCH_SIZE = 64
#NUM_EPOCHS = 3

#1. Transform and load data
data_transforms = data_transforms()
image_datasets = image_datasets(in_arg['dir'], data_transforms)
data_loaders = data_loaders(image_datasets)

#2. Get device for training
if in_arg['gpu'] == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using {device} device for training/validating")

#3. Build/Customize/initialize the Neural Network model and create an instance of it
if in_arg['arch_type'] == 'existing':
    model_ft, input_size = initialize_existing_models(in_arg['arch'], in_arg['arch_type'], NUM_CLASSES, in_arg['feature_extract'],
                                                      in_arg['hidden_units'], use_pretrained=True)
elif in_arg['arch_type'] == 'custom':
    model_ft = build_custom_models(in_arg['arch'], in_arg['arch_type'], NUM_CLASSES, in_arg['feature_extract'], 
                                   in_arg['hidden_units'], use_pretrained=True)
elif in_arg['arch_type'] == 'new':
    print("New neural network models from scratch not yet supported, please use either existing or custom architecture type, exiting...")
    exit()
else:
    print("Invalid or no architecture type selected, please use either existing or custom architecture type, exiting...")
    exit()
#Display model layers for chosen architecture
if in_arg['arch'] == 'resnet18':
    print(model_ft.fc)
else:    
    print(model_ft.classifier)
model_ft = model_ft.to(device) # Send the model to GPU

#4. Create Optimizer - #Gather parameters to update
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
optimizer_ft = optim.Adam(params_to_update, lr=in_arg['learning_rate']) #, momentum=0.9 ##Potential Hyperparameter

#5. Set Loss criterion
criterion = nn.NLLLoss() #nn.CrossEntropyLoss() #Potential Hyperparameter

#6. Train and Validate
model_ft, hist = train_model(model_ft, data_loaders, criterion, optimizer_ft, num_epochs=in_arg['epochs'], device=device,
                             is_inception=(in_arg['arch']=="inception"))

#.7 Save the model
checkpoint = {'arch': in_arg['arch'],
              'arch_type': in_arg['arch_type'],
              'class_to_idx' : image_datasets['train'].class_to_idx,
              'state_dict': model_ft.state_dict(),
              'hidden_units': in_arg['hidden_units'],
              'num_classes' : NUM_CLASSES,
              'feature_extract' : in_arg['feature_extract'],
              'gpu_or_cpu' : in_arg['gpu']
             }
#torch.save(checkpoint, in_arg['save_dir'] + 'checkpoint-' + in_arg['arch'] + '.pth')
torch.save(checkpoint, '../checkpoints/checkpoint.pth')
print(f"Saved checkpoint as {in_arg['save_dir']}checkpoint.pth")
print("***")