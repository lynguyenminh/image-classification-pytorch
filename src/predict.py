from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import yaml

cudnn.benchmark = True
plt.ion()

from utils.load_model import load_model
from utils.load_data import data_transforms

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--numclass', type=int, required=True)
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    return args


def predict(model, dataloaders):
    predict_list = []
    for inputs, _ in dataloaders:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
         
        predict_list += [class_names[i] for i in list(preds.cpu().detach().numpy())]

    list_name_img = [i[0] for i in image_datasets.imgs]
    result = np.concatenate((np.array(list_name_img).reshape(-1, 1), np.array(predict_list).reshape(-1, 1)), axis=1)

    df = pd.DataFrame(data=result, columns=['fname', 'predict_class'])
    df.to_csv('../predict.csv', index=False)

args = get_parse()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. read class_names
with open('config.yaml', "r") as f:
    config = yaml.safe_load(f)
class_names = config['CLASSNAME']


# load model
model = load_model()
model = model.to(device)
model.eval()


# 2. check xem image or folder
if os.path.isdir(args.test_path): 
    image_datasets = datasets.ImageFolder(args.test_path, data_transforms['val'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=8, shuffle=False, num_workers=4)
    
    predict(model, dataloaders)
    print('Predict complete! Result is saved in predict.csv.')

elif os.path.isfile(args.test_path):
    img = Image.open(args.test_path)
    img = data_transforms['val'](img)
    img = img.reshape((1, 3, 224, 224))

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        
    predict_list = [class_names[i] for i in list(preds.cpu().detach().numpy())]
    print(f'Model predict class: {predict_list[0]}')
else: 
    print('Test data path not found!')