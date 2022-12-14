from __future__ import print_function, division
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
plt.ion()

import warnings
warnings.filterwarnings('ignore')



def load_optimization(model=None, optim_name='Adam',lr=0.001):
    if optim_name=="Adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=lr)
    elif optim_name=="RAdam":
        optimizer_ft = optim.RAdam(model.parameters(), lr=lr)
    elif optim_name=="SGD":
        optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_name=="Adadelta":
        optimizer_ft = optim.Adadelta(model.parameters(), lr=lr)
    elif optim_name=="Adagrad":
        optimizer_ft = optim.Adagrad(model.parameters(), lr=lr)
    elif optim_name=="AdamW":
        optimizer_ft = optim.AdamW(model.parameters(), lr=lr)
    elif optim_name=="Adamax":
        optimizer_ft = optim.Adamax(model.parameters(), lr=lr)
    elif optim_name=="ASGD":
        optimizer_ft = optim.ASGD(model.parameters(), lr=lr)
    elif optim_name=="NAdam":
        optimizer_ft = optim.NAdam(model.parameters(), lr=lr)
    elif optim_name=="Rprop":
        optimizer_ft = optim.Rprop(model.parameters(), lr=lr)

    return optimizer_ft