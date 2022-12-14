from __future__ import print_function, division

import os
import matplotlib.pyplot as plt
import yaml

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

cudnn.benchmark = True
plt.ion()

import warnings
warnings.filterwarnings('ignore')


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def load_data(data_dir="../data", batchsize=16, num_workers=4):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # ghi classnane vao config
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    config['CLASSNAME'] = class_names
    with open('config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
    return dataloaders, dataset_sizes, class_names
