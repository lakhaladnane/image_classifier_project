import argparse
# importing Path function from pathlib to use to dynamically add the full path to this directory as to have it work on local machine
from pathlib import Path
import torch

import utility as util
from model_functions import model_builder, train_model, save_checkpoint

main_dir = str(Path(__file__).parent.absolute())

def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default="flowers", help='path to the folder of images')
    parser.add_argument('--save_dir', type=str, default = main_dir + '/saved_models', help='path to the directory where checkpoint will be saved')
    parser.add_argument('--arch', type = str, choices = ['densenet121', 'vgg19'], default = 'densenet121', help='name of pretrained CNN model to use')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help='learning rate used for optimezer')
    parser.add_argument('--hidden_units', type = int, help='number of hidden units')
    parser.add_argument('--epochs', type = int, default = 10, help='number of epochs to use in training the model')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train the model')
    args = parser.parse_args()

    return args

args = init_argparse()

# set device to cpu unless gpu specified
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

# Get images for training
image_data = util.load_data(args.data_dir)

# Create the model based on command line options entered
model = model_builder(args.arch, args.hidden_units)

# train the model
train_model(model, image_data, device, args.learning_rate, args.epochs)

# save the model
save_checkpoint(args.arch, model, image_data, args.save_dir)








