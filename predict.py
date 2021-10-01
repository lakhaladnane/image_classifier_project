import argparse
# importing Path function from pathlib to use to dynamically add the full path to this directory as to have it work on local machine
from pathlib import Path
import torch
from model_functions import load_checkpoint
from utility import predict, cat_to_name

main_dir = str(Path(__file__).parent.absolute())

def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="path to image used for prediction")
    parser.add_argument('checkpoint', help='name of checkpoint to load')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train the model')
    parser.add_argument('--top_k', default=5, type=int, help='set the top k number of classes and their percent match')
    parser.add_argument('--category_names', default='cat_to_name.json', help='json file to use for mapping of categories to real flower names')
    args = parser.parse_args()
    return args

args = init_argparse()
# set device to cpu unless gpu specified
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
image_path = main_dir + '/' + args.input
checkpoint = 'saved_models/' + args.checkpoint
model = load_checkpoint(checkpoint, device)
cat_to_name = cat_to_name(args.category_names)
probs, classes = predict(image_path, model, args.top_k, device)
probs = probs.cpu().numpy().squeeze()
classes = classes.cpu().numpy().squeeze()
flower_names = [cat_to_name[str(i+1)] for i in classes]

def print_result():
    index=1
    for prob, flower in zip(probs, flower_names):
        print(f"{index}. Flower Name: {flower} --", f"Probability: {prob * 100:.2f}%")
        index += 1

print_result()


