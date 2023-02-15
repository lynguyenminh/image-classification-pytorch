import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from utils.load_config import load_config, save_config

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')   


def load_data():
    # load config
    config = load_config('config.yaml')
    IMG_SIZE = config['DATA']['IMG_SIZE'] if config['DATA']['IMG_SIZE'] else (224, 224)
    DATA_DIR = config['DATA']['DATA_DIR'] if config['DATA']['DATA_DIR'] else '../data'
    BATCHSIZE = config['DATA']['BATCHSIZES'] if config['DATA']['BATCHSIZES'] else 16
    NUM_WORKERS = config['DATA']['NUM_WORKERS'] if config['DATA']['NUM_WORKERS'] else 4


    # declare transforms for dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE), # resize anh
            transforms.RandomAdjustSharpness(5.0), #sharpen image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE), # resize anh
            transforms.RandomAdjustSharpness(5.0), #sharpen image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCHSIZE,
                                                shuffle=True, num_workers=NUM_WORKERS)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    # update config.yaml
    config['CLASSNAME'] = class_names
    config['DATA']['IMG_SIZE'] = IMG_SIZE
    config['DATA']['DATA_DIR'] = DATA_DIR
    config['DATA']['BATCHSIZES'] = BATCHSIZE
    config['DATA']['NUM_WORKERS'] = NUM_WORKERS
    save_config(config, 'config.yaml')

        
    return dataloaders, dataset_sizes, class_names

