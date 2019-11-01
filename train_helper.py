import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision


# Define helper functions for training
# Class to create a network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()

        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


# Function to implement a training pass on a network
def train(model, trainloader, testloader, criterion, optimizer, epochs, train_with_gpu, print_step=5):
    # Use GPU if it's available and move model to the right device
    if train_with_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Cuda is not available hence we are switching to default option of CPU")
            device = "cpu"
    else:
        device = "cpu"

    print("This is the device the model is being trained on:", device)
    model.to(device)

    # Keep track of the number of steps taken
    num_steps = 0

    for epoch in range(epochs):
        # Set running loss to keep track of the loss after every printed step
        running_loss = 0

        for inputs, labels in trainloader:
            num_steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear optimizers
            optimizer.zero_grad()

            # Obtain the logarithmic output (Forward Pass)
            logps = model.forward(inputs)

            # Obtain the loss
            loss = criterion(logps, labels)

            # Perform backward propagation
            loss.backward()

            # Take an optimizer step
            optimizer.step()

            # Update the loss so far.
            running_loss += loss.item()

            if num_steps % print_step == 0:
                # Initialize variables to keep track of training loss and testing loss
                test_loss = 0
                accuracy = 0

                # Set model to evaluation mode so gradients are not updated.
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_step:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")

                running_loss = 0

                # Set model back to evaluation mode so gradients are considered.
                model.train()


# Function to implement a validation pass of a model on a dataset
def validation_pass(model, validationloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in validationloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss / len(validationloader):.3f}.. "
          f"Test accuracy: {accuracy / len(validationloader):.3f}")
