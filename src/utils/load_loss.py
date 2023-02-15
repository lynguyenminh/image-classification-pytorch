import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.load_config import load_config, save_config


cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')





def load_loss_function():
    # load config
    config = load_config('config.yaml')
    LOSS_FUNCTION = config['MODEL']['LOSS_FUNCTION'] if config['MODEL']['LOSS_FUNCTION'] else 'CrossEntropyLoss'


    try: 
        if LOSS_FUNCTION == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif LOSS_FUNCTION == "NLLLoss":
            criterion = nn.NLLLoss()
        
        # update config.
        config['MODEL']['LOSS_FUNCTION'] = LOSS_FUNCTION
        save_config(config, 'config.yaml')

    except:
        print('Error: Could not find loss function. Please check loss function name.')
        exit(1)
    return criterion