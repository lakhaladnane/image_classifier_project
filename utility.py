import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
import json


def load_data(image_dir):
    """
    Retrieve images from path given to image folder which will be used to train the model

    images_dir: 
      relative path to the folder of images that are to be
      classified by the classifier function (string).
      The Image folder is expected to have three sub folders:
        - train
        - valid
        - test
    """
    image_dir = image_dir
    train_dir = image_dir + '/train'
    valid_dir = image_dir + '/valid'
    test_dir = image_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    image_dict = {  "train_data": train_data,
                    "valid_data": valid_data,
                    "test_data": test_data,
                    "trainloader": trainloader,
                    "validloader": validloader,
                    "testloader": testloader
                }
    return image_dict

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    
    Parameters:
      image_dir:
        path to image to be processed
    
    Returns:
      a Numpy array
    """
    
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # Resize the image and get the new dimensions to use in crop below
    img.thumbnail((256,256))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (left + 224)
    bottom = (top + 224)
    
    # Crop the center of the image to 224 x 224 dimension
    img = img.crop((left, top, right, bottom))
    np_image = np.array(img)
    np_image = np_image.astype('float32') / 255.0
    
    # Normalize images in specific way needed for network
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose image array to fit PyTorch expeted shape with colour channel as first dimension
    np_image = np_image.transpose((2,0,1))
    image_tensor = torch.from_numpy(np_image)
    return image_tensor

def cat_to_name(file):
    """
    Loads a json file with mapping from category to flower name

    Parameters:
      file:
        name of .json mapping file

    Returns:
      a python dictionary with mapping of categories to flower names
    """
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def predict(image_path, model, topk, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Parameters:
      image_path:
        path to image to predict
      model:
        CNN model to use to make prediction
      topk:
        The number of results you wish to be printed
      device:
        Either "cpu" or "cuda".
    """
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    image = image.to(device)
    model = model
    model.eval()
    model.to(device)
    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        return ps.topk(topk, dim=1)