# -*- coding: utf-8 -*-

#User Input
SOURCE_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
BASE_DIRECTORY = 'flowers'
DATA_DIRECTORY = 'data'
MODEL_DIRECTORY = 'models'
FILE_NAME = 'flowers.tar.gz'
MODELS_REPO = 'flower-models-repo'

#Multiple Experiment Management
list_of_num_epochs = [5, 10]
list_of_models_and_weights = [
    # ('mobilenet_v2','MobileNet_V2_Weights'),
    # ('densenet121','DenseNet121_Weights'),
    # ('inception_v3','Inception_V3_Weights'),
    # ('efficientnet_b2','EfficientNet_B2_Weights'),
    # ('squeezenet1_1','SqueezeNet1_1_Weights'),
    # ('vgg16','VGG16_Weights'),
    # ('alexnet','AlexNet_Weights'),
    # ('resnet18','ResNet18_Weights'),
    # ('swin_b','Swin_B_Weights'),
    ('vit_b_16', 'ViT_B_16_Weights')
                             ]
list_of_sample_sizes = [0.5, 1.0] #[0.33, 0.66, 1.0]     
list_of_loss_functions = ['CrossEntropyLoss'] 
list_of_optimizers = ['Adam']                      
list_of_learning_rate = [3e-3]

#Generally not required to be changed during training
BATCH_SIZE = 64
NUM_CLASSES = 102
FEATURE_EXTRACT = True
RGB = 3 #(Color picture is 3, black & white is 1)
MANUAL_RESIZE = 64 #Not used
HIDDEN_UNITS = '' #Not used for pretrained models, will add later

# Setup hyperparameters
# MODEL_NAME = 'vit_b_16'
# MODEL_WEIGHT = 'ViT_B_16' 
# NUM_EPOCHS = 10
# LEARNING_RATE = 3e-3
# LOSS_FUNCTION = 'CrossEntropyLoss'
# OPTIMIZER = 'Adam'

"""###Get Libraries"""

# Install atleast torch 1.12+ and torchvision 0.13+
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"Using torch version: {torch.__version__}")
    print(f"Using torchvision version: {torchvision.__version__}")
except:
    #print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    #!pip3 install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

"""###Regular Imports"""

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

try:
    from torchinfo import summary # Try to get torchinfo, install it if it doesn't work
except:
    #print("[INFO] Couldn't find torchinfo... installing it.")
    #!pip install -q torchinfo
    from torchinfo import summary

#Additions from functions
import os
import time
import sys
import tarfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

"""###Helpers/functions from Github"""

# Try to import the helper functions, download it from GitHub if it doesn't work
try:
    sys.path.append(os.path.join(os.getcwd(), BASE_DIRECTORY))    
    import data_setup, engine, model_builder, utils 
    from helper_functions import download_data, set_seeds, plot_loss_curves, create_directory  
    from predictions import pred_and_plot_image, create_writer
except:
    # Get the scripts
    print("[INFO] Couldn't find the scripts... downloading them from GitHub.")
#     !git clone https://github.com/rdkworld/AIPND-2022
#     #create_directory(Path().absolute() / BASE_DIRECTORY)
#     !mkdir --parents /content/$BASE_DIRECTORY 
#     !mv AIPND-2022/Generalized/*.py /content/$BASE_DIRECTORY
#     !rm -rf AIPND-2022
    sys.path.append(os.path.join(os.getcwd(), BASE_DIRECTORY))
    import data_setup, engine, model_builder, utils 
    from helper_functions import download_data, set_seeds, plot_loss_curves, create_directory, create_writer
    from predictions import pred_and_plot_image

#Create Directory Structure
create_directory(Path(BASE_DIRECTORY))
create_directory(Path(BASE_DIRECTORY) / DATA_DIRECTORY)
create_directory(Path(BASE_DIRECTORY) / MODEL_DIRECTORY)

train_dir = f"{BASE_DIRECTORY}/{DATA_DIRECTORY}/{BASE_DIRECTORY}/train"
valid_dir = f"{BASE_DIRECTORY}/{DATA_DIRECTORY}/{BASE_DIRECTORY}/valid"
test_dir = f"{BASE_DIRECTORY}/{DATA_DIRECTORY}/{BASE_DIRECTORY}/test"

"""###Setup target device"""

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
device

"""##Download data and categorize into train/valid/test folders as required"""
"""## Ignore below errors, data is getting unzipped, need to figure out to remove the error though"""
#Untar the file
if not os.path.exists(Path(BASE_DIRECTORY) / DATA_DIRECTORY / BASE_DIRECTORY):
    with tarfile.open(Path(MODELS_REPO) / FILE_NAME, "r") as tar_ref:
        print(f"[INFO] Unzipping {FILE_NAME}...")
        try: 
          tar_ref.extractall(Path(BASE_DIRECTORY) / DATA_DIRECTORY)
        except EOFError:
          print("Bypassing EOFError but files unzipped")

    if (Path(BASE_DIRECTORY) / DATA_DIRECTORY / FILE_NAME).is_file():
      (Path(BASE_DIRECTORY) / DATA_DIRECTORY / FILE_NAME).unlink()

"""#Experiment Loop starts"""
experiment_number = 0

#loop 1 - Loop thru different models, if experimenting
prior_model = ''
for model in list_of_models_and_weights:
  #Get pre-trained model weights and model
  MODEL_NAME, MODEL_WEIGHT = model
  pretrained_weights = eval(f"torchvision.models.{MODEL_WEIGHT}.DEFAULT")
  pretrained_model = eval(f"torchvision.models.{MODEL_NAME}(weights = pretrained_weights)").to(device)
  auto_transforms = pretrained_weights.transforms()

  # Create model with help from model_builder.py
  updated_pretrained_model = model_builder.update_last_layer_pretrained_model(pretrained_model, NUM_CLASSES, FEATURE_EXTRACT).to(device)

  #Loop 2 - Loop thru different sizes of input dataset, if experimenting
  test_accuracy_of_best_experiment = 0.0
  for sample in list_of_sample_sizes:
    if list_of_sample_sizes and 1.0 not in list_of_sample_sizes:
      # Create data loaders
      train_dataloader, test_dataloader, class_names, class_to_idx = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                   test_dir=test_dir,
                                                                                                   transform=auto_transforms,
                                                                                                   batch_size=BATCH_SIZE,
                                                                                                   sample_size = sample
                                                                                                   )
    else:
      train_dataloader, test_dataloader, class_names, class_to_idx = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                   test_dir=test_dir,
                                                                                                   transform=auto_transforms,
                                                                                                   batch_size=BATCH_SIZE
                                                                                                   )
      sample = 1.0
    if len(class_names) != NUM_CLASSES:
      print("Mismatch in the number of unique classes/labels and user input NUM_CLASSES")
      exit()

    #Loop 3 - Loop thru different loss function, if experimenting
    for lf in list_of_loss_functions:
      loss_fn = eval(f"torch.nn.{lf}()")
      #Loop 4 - Loop thru different optimizers, if experimenting
      for optim in list_of_optimizers:
        #Loop 5 - Loop thru learning rates, if experimenting
        for lr in list_of_learning_rate:
          optimizer = eval(f"torch.optim.{optim}(updated_pretrained_model.parameters(),lr=lr)")
          #Loop 6 - Loop thru different set of epochs, , if experimenting
          for num_epochs in list_of_num_epochs:
            #Finally, do the training 
            since = time.time()
            experiment_number += 1
            dict_for_writer = {}
            dict_for_writer = {'experiment_number' : f"Run_{experiment_number}" ,
                               'model_name' : MODEL_NAME,
                               'sample_size' : f"{sample*100}% Data" if len(list_of_sample_sizes) > 1 else None,
                               'loss_fn' : f"lossfunction_{lf}" if len(list_of_loss_functions) > 1 else None,
                               'optimizer' : f"optimizer_{optim}" if len(list_of_optimizers) > 1 else None,
                               'learning_rate' : f"lr_{lr}" if len(list_of_learning_rate) > 1 else None,
                               'num_epochs' : f"{num_epochs}_epochs"  if len(list_of_num_epochs) > 1 else None
                              }                             
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"EXPERIMENT {experiment_number} STARTS | MODEL is {MODEL_NAME} | SAMPLE SIZE {sample} | LOSS FUNCTION {lf} | OPTIMIZER {optim} | LEARNING RATE {lr} | NUM EPOCHS {num_epochs}")
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            pretrained_model_results = engine.train(model=updated_pretrained_model,
                                                    train_dataloader=train_dataloader,
                                                    test_dataloader=test_dataloader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    epochs=num_epochs,
                                                    device=device,
                                                    writer=create_writer(dict_for_writer = dict_for_writer)             
                                                    )                    
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            time_elapsed = time.time() - since
            
            #Determine in-memory model-size
            param_size = 0
            for param in updated_pretrained_model.parameters():
                param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in updated_pretrained_model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / 1024**2
            model_size = f"{size_all_mb:.1f} MB" 
            print("Calculated best model size based on parameters and buffers ", model_size) 
            
            print('EXPERIMENT {} | Completed in {:.0f} min {:.0f}sec | TEST ACCURACY {} | MODEL SIZE {}'.format(experiment_number, time_elapsed // 60, time_elapsed % 60, pretrained_model_results['test_acc'][-1], model_size))
            training_time = f"{time_elapsed // 60:.0f} min {time_elapsed % 60:.0f} secs"           

            if (MODEL_NAME == prior_model and pretrained_model_results['test_acc'][-1] > test_accuracy_of_best_experiment) or MODEL_NAME != prior_model:
              prior_model = MODEL_NAME
              test_accuracy_of_best_experiment = pretrained_model_results['test_acc'][-1]
              print(f"SAVING THIS EXPERIMENT {experiment_number} | MODEL is {MODEL_NAME} | SAMPLE SIZE {sample} | LOSS FUNCTION {lf} | OPTIMIZER {optim} | LEARNING RATE {lr} | NUM EPOCHS {num_epochs}")
           
              #Save the model
              checkpoint = {'state_dict': updated_pretrained_model.state_dict(),
                            'arch': MODEL_NAME,
                            'arch_weight': MODEL_WEIGHT,              
                            'arch_type': 'EXISTING',
                            'loss_function': lf,
                            'optimizer': optim,                            
                            'class_names' : class_names,
                            'class_to_idx' : class_to_idx,
                            'hidden_units': HIDDEN_UNITS,
                            'num_classes' : NUM_CLASSES,
                            'feature_extract' : FEATURE_EXTRACT,
                            'gpu_or_cpu' : device,
                            'num_epochs' : num_epochs,
                            'learning_rate' : lr,
                            'loss_function' : lf,
                            'optimizer' : optim,
                            'batch_size' : BATCH_SIZE,
                            'training_time' : training_time,
                            'model_size' : model_size
                          }       
              print(f"Checkpoint for {MODEL_NAME} is below")
              # Save the model on local drive
              m_name = f"{BASE_DIRECTORY}_{MODEL_NAME}_model.pth"
              utils.save_model(model=updated_pretrained_model, target_dir=f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}", model_name=m_name, checkpoint = checkpoint)

              #Size of the model
              model_file = f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{m_name}"
              pretrained_model_size = Path(model_file).stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly) 
              print(f"Best model size based on size on disk: {pretrained_model_size} MB")                             

print("Ending")
exit()
print("After Exit")

"""##Load the model

"""

#Check the device and load the checkpoint
if torch.cuda.is_available():
    device = torch.device("cuda") 
    checkpoint1 = torch.load(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{MODEL_NAME}_model.pth")
else:
    device = "cpu" #or torch.device("cpu") 
    checkpoint1 = torch.load(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{MODEL_NAME}_model.pth", map_location = device)
print(f"Using {device} device for predicting/inference")

#Load/initialize the model
pretrained_weights1 = eval(f"torchvision.models.{MODEL_WEIGHT}.DEFAULT")
auto_transforms1 = pretrained_weights1.transforms()
pretrained_model1 = eval(f"torchvision.models.{MODEL_NAME}(weights = None)")
pretrained_model1 = model_builder.update_last_layer_pretrained_model(pretrained_model1, NUM_CLASSES, FEATURE_EXTRACT) 
pretrained_model1.class_to_idx = checkpoint1['class_to_idx']
pretrained_model1.class_names = checkpoint1['class_names']
pretrained_model1.load_state_dict(checkpoint1['state_dict'])
pretrained_model1.to(device)

"""##Inference"""

# !ls flowers/data/flowers/test/*

# Predict on custom image
#updated_pretrained_model
custom_image_path = 'flowers/data/flowers/test/96/image_07622.jpg'
pred_and_plot_image(model=pretrained_model1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=auto_transforms)