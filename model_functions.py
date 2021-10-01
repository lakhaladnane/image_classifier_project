import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import keep_awake


def model_builder(arch, hidden_units):
    """
    Build model based on command line inputs

    Parameters:
      arch:
        The model architecture to use from torchvision. The two options are 'densenet121' or 'vgg19'
      hidden_units:
        The number of hidden units to use in the classifier layer

    Returns:
        CNN model with custom classifier feed forward layer.
    """
    
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters to not backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = None

    if arch == "densenet121" and hidden_units is None:
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 512)),
                                ('relu1', nn.ReLU()),
                                ('drp1', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(512, 256)),
                                ('relu2', nn.ReLU()),
                                ('drp2', nn.Dropout(0.5)),
                                ('fc3', nn.Linear(256, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    elif arch == "densenet121" and hidden_units is not None:
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 512)),
                                ('relu1', nn.ReLU()),
                                ('drp1', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(512, hidden_units)),
                                ('relu2', nn.ReLU()),
                                ('drp2', nn.Dropout(0.5)),
                                ('fc3', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    elif arch == "vgg19" and hidden_units is None:
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu1', nn.ReLU()),
                            ('drp1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, 2043)),
                            ('relu2', nn.ReLU()),
                            ('drp2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(2043, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    else:
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu1', nn.ReLU()),
                            ('drp1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('drp2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier
 
    return model

def train_model(model, image_data, device, learning_rate, epochs):
    """
    Traines the given model on given image data directory using user input parameters through command line

    Parameters:
      model:
        The model to train
      image_data:
        The path to the folder with all the images to use to train the model. This folder needs to have three sub folders
        (test, train, valid) with images already split up in each sub folders to be used to train, validate and test model accuracy.
      device:
        Either "cpu" or "cuda".
      learning_rate:
        learning rate to pass to Adam optimizer.
      epochs:
        number of epochs to use in training

    Returns:
        None - model is passed in and trained based on hyperparameters passed along with  model

    """

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = epochs
    model.to(device)

    train_losses, validation_losses = [], []
    for e in keep_awake(range(epochs)):
        running_loss = 0
        for images, labels in image_data["trainloader"]:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            validation_loss = 0
            accuracy = 0
            # set model to evaluation mode
            model.eval()
            with torch.no_grad():
                
                for images, labels in image_data["validloader"]:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    validation_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss/len(image_data["validloader"]))
            validation_losses.append(validation_loss/len(image_data["validloader"]))
            
            print(f"Epoch: {e+1}/{epochs}.. ",
                f"Training Loss: {running_loss/len(image_data['trainloader']):.3f}.. ",
                f"Validation Loss: {validation_loss/len(image_data['validloader']):.3f}.. ",
                f"Validation Accuracy: {accuracy/len(image_data['validloader']):.3f}")
            
            # set model back to train mode
            model.train()

def save_checkpoint(arch, model, image_data, save_dir):
    """
    Save a checkpoint.pth file after model is trained in order to be able to later load this model without needing to train again

    Parameters:
      arch:
        The model architecture to use from torchvision. The two options are 'densenet121' or 'vgg19'
      model:
        trained model to save as checkpoint
      image_data:
        image_data dictionary
      save_dir:
        directory to save the checkpoint
    """
    checkpoint = {
                'arch': arch,
                'classifier': model.classifier ,
                'state_dict': model.state_dict(),
                'class_to_idx': image_data['train_data'].class_to_idx
             }
    save_location = save_dir + '/checkpoint2.pth'
    torch.save(checkpoint, save_location)

def load_checkpoint(filepath, device):
    """
    load checkpoint from pre-trained model

    Parameters:
      filepath:
        path to checkpoint to load
      device:
        Either "cpu" or "cuda".
    """
    checkpoint = torch.load(filepath, map_location=device)
    model = getattr(models, checkpoint["arch"])(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    
    return model