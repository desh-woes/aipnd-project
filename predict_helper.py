# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image
import json

import train_helper

# Function to read categories from a file
def read_category(category_dir):
    with open(category_dir, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

# Done: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # Load pre-trained model
    model = getattr(models, checkpoint["model_arch"])(pretrained=True)

    # Freze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = train_helper.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint["class_to_idx"]


# Function to process input Image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load the image
    img = Image.open(image_path)
    
    # Get image dimensions
    currWidth, currHeight = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension so the shortest size is 255px
    if currWidth < currHeight:
        newHeight = int(currHeight*256/currWidth)
        img.resize((256, newHeight))
    else:
        newWidth = int(currWidth*256/currHeight)
        img.resize((newWidth, 256))
        
    newWidth, newHeight = img.size
    
    # Center crop the image
    left = (newWidth - 224)/2
    top = (newHeight - 224)/2
    right = (newWidth + 224)/2
    bottom = (newHeight + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(img)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
    # Transpose Image
    img = np_image.transpose((2, 0, 1))
    
    
    # Convert image to a tensor.
    image_tensor = torch.from_numpy(img).float()
    
    return image_tensor

# Function to make predictions
def predict(img_tensor, model, class_to_idx, topk, train_with_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if train_with_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Cuda is not available hence we are switching to default option of CPU")
            device = "cpu"
    else:
        device = "cpu"
    
    img_tensor = img_tensor[np.newaxis,:]
    print("This is the device in use", device)
    
    # Move model and image to device
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    
    # Pass image through the model
#     class_to_idx = train_data.class_to_idx
    logps = model.forward(img_tensor)

    # Get prediction
    ps = torch.exp(logps)
    
    probs, indices = ps.topk(topk, dim=1)
    probs = probs.cpu().detach().numpy()[0]
    indices  = indices.cpu().detach().numpy()[0]

    idx_to_class = {v:k for k, v in class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    return probs, classes
    