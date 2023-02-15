from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True
plt.ion()

from utils.load_model import load_model
from utils.load_data import load_data
from utils.load_optim import load_optimization
from utils.load_loss import load_loss_function
from utils.train_model import train_model
from utils.load_config import load_config, save_config


# def get_parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, required=True)
#     parser.add_argument('--data', type=str, required=True)
#     parser.add_argument('--checkpoint', type=str, required=False)
#     parser.add_argument('--epochs', type=int, required=True)
#     parser.add_argument('--batchsize', type=int, required=True)
#     parser.add_argument('--numclass', type=int, required=True)
#     parser.add_argument('--save_weights', type=str, required=True)
#     args = parser.parse_args()
#     return args

from utils.load_config import load_config, save_config

# config = load_config('config.yaml')

# LOSS_FUNCTION = config['MODEL']['LOSS_FUNCTION'] if config['MODEL']['LOSS_FUNCTION'] else 'CrossEntropyLoss'



if __name__=="__main__":
    # args = get_parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataloaders, dataset_sizes, class_names = load_data()

    # load model
    model = load_model()
    model = model.to(device)

    # load loss function
    criterion = load_loss_function()

    # load optimization function
    optimizer_ft = load_optimization(model=model)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # train model
    train_model(model=model, device=device, dataloaders=dataloaders, 
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler)
    
