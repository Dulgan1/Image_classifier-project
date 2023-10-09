#!/usr/bin/env python3
"""Image Classifier Terminal app: Build, Train, Validate and Test Model
Usage:
python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 10 --save_dir ./ --arch vgg13 --gpu
"""
import torch
import json
import numpy as np
from torch import nn
from torch import optim
import argparse
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time

# Parser object named parser, for parsing argument
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default='flowers', help='Provide the data directory, mandatory')
parser.add_argument('--save_dir', type = str, default = './', help = 'Provide the save directory')
parser.add_argument('--arch', type = str, default = 'densenet121', help = 'densenet121 or vgg13')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help = 'Learning rate; default is 0.001')
parser.add_argument('--hidden_units', type=int, default=512,
                    help='Number of hidden units. Default value is 512')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help="Add to activate CUDA")
args_in = parser.parse_args()

if args_in.gpu:
    device = torch.device("cuda")
    print("~~~~~~~~~~~~~~ [CUDA initialized] ~~~~~~~~~~~~~~~~")
else:
    device = torch.device("cpu")

print("########## [Loading data...] ##########")

data_dir  = args_in.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

# DONE: Define your transforms for the training, validation, and testing sets
data_train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_test_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# DONE: Load the datasets with ImageFolder
image_train_datasets = datasets.ImageFolder(train_dir, transform=data_train_transforms)
image_valid_datasets = datasets.ImageFolder(valid_dir, transform=data_valid_transforms)
image_test_datasets  = datasets.ImageFolder(test_dir, transform=data_test_transforms)

# DONE: Using the image datasets and the trainforms, define the dataloaders
dataloader_train = torch.utils.data.DataLoader(image_train_datasets, batch_size = 64, shuffle = True)
dataloader_valid = torch.utils.data.DataLoader(image_valid_datasets, batch_size = 64)
dataloader_test  = torch.utils.data.DataLoader(image_test_datasets,  batch_size = 64)


# Label mapping from json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print('########## [Data loaded!!!] ##########\n\n\n')


# Build Image Classifier Model

print("~~~~~~~~~~~~~~ [Building Model!!!] ~~~~~~~~~~~~~~")

layers = args_in.hidden_units
learning_rate = args_in.learning_rate

if args_in.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
elif args_in.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
else:
    raise ValueError('Model arch error.')

criterion = nn.NLLLoss()
optimiz = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);

print("[~~~~~~~~ [Model Architecture: " + args_in.arch + "] ~~~~~~~~~~]")
print("~~~~~~~~~~~~[ [Model Build Finished!!!] ]~~~~~~~~~~~~~")


# Train Model

print("~~~~~~~~~~~~ [Training Model...] ~~~~~~~~~~~~~")

running_loss = 0
epochs = 30
print_step = 5
steps = 0
valid_acc = 0.0

for epoch in range(epochs):
    init_time = time()
    for inputs, labels in dataloader_train:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimiz.zero_grad()
        
        log_ = model.forward(inputs)
        loss  = crtn(log_, labels)
        loss.backward()
        optimiz.step()
        
        running_loss += loss.item()
        
        if steps % print_step == 0:
            accuracy = 0
            loss_test = 0
            
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in dataloader_valid:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ = model.forward(inputs)
                    batch_loss = crtn(log_, labels)
                    loss_test += batch_loss.item()
                    
                    ps = torch.exp(log_)
                    
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print("(Epoch {}/{}) ... Train loss: [{:.2f}]... Validation accuracy: [{:.2f}]...\
            Validation loss: [{:.3f}]".\
                  format(epoch+1, epochs, running_loss/print_step, accuracy/len(dataloader_valid),
                         loss_test/len(dataloader_valid)))        
            running_loss = 0
            model.train()
            valid_acc = accuracy/len(dataloader_valid)
    if valid_acc > 0.87:
        break
    final_time = time()
    print("Epoch Elapsed Runtime {}: {}s.".format(epoch, final_time-init_time))

print("~~~~~~~~~~~ [MODEL TRAINED] ~~~~~~~~~~~")


# Test Model

print("~~~~~~~~~~~~ [Testing The Model] ~~~~~~~~~~~~~~~")
model.to(device);
accuracy =0
model.eval()

with torch.no_grad():
    for inputs, labels in dataloader_test:
        inputs, labels = inputs.to(device), labels.to(device)
        log_ = model.forward(inputs)
        ps_ = torch.exp(log_)
        top_p, top_class = ps_.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        

print("TEST ACCURACY [{:.3f}]".format(accuracy/len(dataloader_test)))
print("~~~~~~~~~~~~~ [Model Testing Done] ~~~~~~~~~~~~~")


# Save checkpoint

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': args_in.arch
             }

save_path = args_in.save_dir + 'checkpoint.pth'
torch.save(checkpoint, save_path)
print("~~~~~~~~~~~~~ [Model Checkpoint Saved] ~~~~~~~~~~~~~~")