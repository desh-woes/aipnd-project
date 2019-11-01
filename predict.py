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

# Import files containing helper functions
import predict_helper

# Setup ArgParser
import argparse

# Create parser object
parser = argparse.ArgumentParser()

# Add different arguments to the parser
parser.add_argument("image_dir", help="Enter Image directory to be predicted")
parser.add_argument("checkpoint", help="Add path to model checkpoint")

# Add optional params
parser.add_argument("--top_k", help="Enter the number of predictions required")
parser.add_argument("--category_names", help="Enter the category name mapping of the classes in the model")
parser.add_argument("--gpu", help="Will you like to predict on gpu", action="store_true", default=False)

# Parse all the arguments
args = parser.parse_args()

# Extract parameters
image_dir = args.image_dir
checkpoint = args.checkpoint
top_k = int(args.top_k) if args.top_k else 1

# Process Image
image = predict_helper.process_image(image_dir)

# Load model from checkpoint
model, class_to_idx = predict_helper.load_checkpoint(checkpoint)

# Give image to model to predict output
top_prob, top_class = predict_helper.predict(image, model, class_to_idx, top_k, args.gpu)

# Check if the category names were provided so category names are printed instead
if args.category_names:
    # Read in the categories
    cat_to_name = predict_helper.read_category(args.category_names)
    new_class = []
    for i in top_class:
        new_class.append(cat_to_name[i])
        
    top_class = new_class

# Print the results
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class)