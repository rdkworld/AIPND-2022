"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
MODEL_NAME = 'TinyVGG'
LOSS_FUNCTION = 'CrossEntropyLoss'
OPTIMIZER = 'Adam'
DATA_DIRECTORY = 'pizza_steak_sushi'
MANUAL_RESIZE = 64

# Setup directories
train_dir = f"data/{DATA_DIRECTORY}/train"
test_dir = f"data/{DATA_DIRECTORY}/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((MANUAL_RESIZE, MANUAL_RESIZE)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.MODEL_NAME(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = eval(torch.nn.LOSS_FUNCTION())
optimizer = torch.optim.OPTIMIZER(model.parameters(),
                             lr=LEARNING_RATE)

# #SINGLE RUN
# #Create Writer
# this_writer = create_writer(experiment_name="data_10_percent",
#                                model_name="effnetb0",
#                                extra="5_epochs")
# # Start training with help from engine.py
# set_seeds()
# engine.train(model=model,
#              train_dataloader=train_dataloader,
#              test_dataloader=test_dataloader,
#              loss_fn=loss_fn,
#              optimizer=optimizer,
#              epochs=NUM_EPOCHS,
#              device=device)

# # Save the model with help from utils.py
# utils.save_model(model=model,
#                  target_dir="models",
#                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")

#MULTIPLE RUNS
%%time

# 1. Set the random seeds
set_seeds(seed=42)

# 2. Keep track of experiment numbers
experiment_number = 0

# 3. Loop through each DataLoader
for dataloader_name, train_dataloader in train_dataloaders.items():

    # 4. Loop through each number of epochs
    for epochs in num_epochs: 

        # 5. Loop through each model name and create a new model based on the name
        for model_name in models:

            # 6. Create information print outs
            experiment_number += 1
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of epochs: {epochs}")  

            # 7. Select the model
            if model_name == "effnetb0":
                model = create_effnetb0() # creates a new model each time (important because we want each experiment to start from scratch)
            else:
                model = create_effnetb2() # creates a new model each time (important because we want each experiment to start from scratch)
            
            # 8. Create a new loss and optimizer for every model
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

            # 9. Train target model with target dataloaders and track experiments
            train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader, 
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  epochs=epochs,
                  device=device,
                  writer=create_writer(experiment_name=dataloader_name,
                                       model_name=model_name,
                                       extra=f"{epochs}_epochs"))
            
            # 10. Save the model to file so we can get back the best model
            save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
            save_model(model=model,
                       target_dir="models",
                       model_name=save_filepath)
            print("-"*50 + "\n")