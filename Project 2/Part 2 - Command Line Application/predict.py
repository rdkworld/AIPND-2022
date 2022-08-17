#Import libraries
import torch
import torchvision.models as models
import json

#Import User Defined libraries
from neural_network_model import initialize_existing_models, build_custom_models, set_parameter_requires_grad
from utilities import process_image, get_input_args_predict

def predict(image_path, model, topk=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    tensor_img = torch.FloatTensor(process_image(image_path))
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device)
    log_ps = model(tensor_img)
    result = log_ps.topk(topk)
    if torch.cuda.is_available(): #gpu Move it from gpu to cpu for numpy
        ps = torch.exp(result[0].data).cpu().numpy()[0] 
        idxs = result[1].data.cpu().numpy()[0]
    else: #cpu Keep it on cpu for nump
        ps = torch.exp(result[0].data).numpy()[0]
        idxs = result[1].data.numpy()[0]

    return (ps, idxs)

#0. Get user inputs
in_arg = vars(get_input_args_predict())
print("User arguments/hyperparameters or default used are as below")
print(in_arg)

#1. Get device for prediction and Load model from checkpoint along with some other information
if in_arg['gpu'] == 'gpu' and torch.cuda.is_available():
    device = torch.device("cuda")
    checkpoint = torch.load(in_arg['save_dir'])
else:
    device = "cpu"
    checkpoint = torch.load(in_arg['save_dir'], map_location = device)
print(f"Using {device} device for predicting/inference")

if checkpoint['arch_type'] == 'existing':
    model_ft, input_size = initialize_existing_models(checkpoint['arch'], checkpoint['arch_type'], len(checkpoint['class_to_idx']),
                                                      checkpoint['feature_extract'], checkpoint['hidden_units'], use_pretrained=False)
elif checkpoint['arch_type'] == 'custom':
    model_ft = build_custom_models(checkpoint['arch'], checkpoint['arch_type'], len(checkpoint['class_to_idx']), checkpoint['feature_extract'], 
                                   checkpoint['hidden_units'], use_pretrained=True)
else:
    print("Nothing to predict")
    exit()
    
    
model_ft.class_to_idx = checkpoint['class_to_idx']
model_ft.gpu_or_cpu = checkpoint['gpu_or_cpu']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.to(device)

#Predict
# Get the prediction by passing image and other user preferences through the model
probs, idxs  = predict(image_path = in_arg['path'], model = model_ft, topk = in_arg['top_k'])

# Swap class to index mapping with index to class mapping and then map the classes to flower category labels using the json file
idx_to_class = {v: k for k, v in model_ft.class_to_idx.items()}
with open('cat_to_name.json','r') as f:
    cat_to_name = json.load(f)
names = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"],idxs))

# Display final prediction and Top k most probable flower categories                               
print("-"*60)
print("                    PREDICTION")
print("-"*60)
print("Image provided        : {}" .format(in_arg['path']))
print("Predicted Flower Name : {} (Class {} and Index {})" .format(names[0].upper(), idx_to_class[idxs[0]], idxs[0] ))
print("Model used            : {}" .format(checkpoint['arch']))
print(f"The top {in_arg['top_k']} probabilities of the flower names")
for name, prob in zip(names, probs):
    length = 30 - len(name)
    print(f"{name.title()}{' '*length}{round(prob*100,2)}%")