import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.load_config import load_config, save_config

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')



def load_optimization(model):
    config = load_config('config.yaml')

    OPTIM_FUNCTION = config['MODEL']['OPTIM_FUNCTION'] if config['MODEL']['OPTIM_FUNCTION'] else 'Adam'
    LEARNING_RATE = config['MODEL']['LEARNING_RATE'] if config['MODEL']['LEARNING_RATE'] else 0.00001


    try: 
        if OPTIM_FUNCTION=="Adam":
            optimizer_ft = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="RAdam":
            optimizer_ft = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="SGD":
            optimizer_ft = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        elif OPTIM_FUNCTION=="Adadelta":
            optimizer_ft = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Adagrad":
            optimizer_ft = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="AdamW":
            optimizer_ft = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Adamax":
            optimizer_ft = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="ASGD":
            optimizer_ft = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="NAdam":
            optimizer_ft = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIM_FUNCTION=="Rprop":
            optimizer_ft = optim.Rprop(model.parameters(), lr=LEARNING_RATE)

        # update config.
        config['MODEL']['OPTIM_FUNCTION'] = OPTIM_FUNCTION
        config['MODEL']['LEARNING_RATE'] = LEARNING_RATE
        save_config(config, 'config.yaml')

    except:
        print('Error: Could not find optim function. Please check optim function name.')
        exit(1)
        
    return optimizer_ft