# Add all required imports from torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision

# Import files containing helper functions
import load_data
import train_helper

# Setup ArgParser
import argparse

# Create parser object
parser = argparse.ArgumentParser()

# Add different arguments to the parser
parser.add_argument("data_directory", help="Enter data_directory")

# Add optional params
parser.add_argument("--learning_rate", help="Enter the prefered learning rate")
parser.add_argument("--hidden_units", help="Enter multiple units seperated with a comma")
parser.add_argument("--epochs", help="Enter Preferred number of training epochs")
parser.add_argument("--gpu", help="Will you like to train model on gpu", action="store_true", default=False)
parser.add_argument("--save_dir", help="Specify the directory where you will like to save the model")
parser.add_argument("--arch", help="Choose an architecture of choice", choices=["densenet121", "densenet161"])


# Parse all the arguments
args = parser.parse_args()

# Obtain compulsory argument
data_dir = args.data_directory
arch = args.arch if args.arch else "densenet121"


# Building training model
# Obtain data loaders
trainloader, testloader, validationloader, class_to_idx = load_data.load_data(data_dir)

# Pre-trained Image model
model = getattr(models, arch)(pretrained=True)


# Freze model parameters
for param in model.parameters():
    param.requires_grad = False
    
print(model)
# Build model classifier and define hyper params
num_input = model.classifier.in_features
num_output = len(class_to_idx)

learning_rate = float(args.learning_rate) if args.learning_rate else 0.003
hidden_layers = list(map(int, args.hidden_units.split(","))) if args.hidden_units else [500, 256]
epochs = int(args.epochs) if args.epochs else 1
gpu = args.gpu
save_dir = args.save_dir if args.save_dir else 'checkpoint.pth'


model.classifier = train_helper.Network(input_size=num_input, output_size=num_output, hidden_layers=hidden_layers)
print("This is the model:..........................................................................")
print(model)

# Define hyper parameters
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Train the network
train_helper.train(model, trainloader, testloader, criterion, optimizer, epochs, gpu)

# Validation with model made directly i.e not loaded from file
train_helper.validation_pass(model, validationloader, criterion)

# Done: Save the checkpoint
checkpoint = {'model_arch': arch,
              'input_size': model.classifier.hidden_layers[0].in_features,
              'output_size': len(class_to_idx),
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
              'state_dict': model.state_dict(),
              'class_to_idx' : class_to_idx
             }

# Save model to a checkpoint
torch.save(checkpoint, save_dir)