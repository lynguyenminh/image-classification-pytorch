from __future__ import print_function, division
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
plt.ion()

import warnings
warnings.filterwarnings('ignore')



def load_loss_function(loss_function_name="CrossEntropyLoss"):
    if loss_function_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name == "NLLLoss":
        criterion = nn.NLLLoss()
    
    return criterion