import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd

cudnn.benchmark = True

from utils.load_config import load_config


config = load_config('config.yaml')
CLASS_NAMES = config['CLASSNAME']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(model, dataloaders):
    predict_list = []
    for inputs in dataloaders[0]:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
         
        predict_list += [CLASS_NAMES[i] for i in list(preds.cpu().detach().numpy())]

    result = np.concatenate((np.array(dataloaders[1]).reshape(-1, 1), np.array(predict_list).reshape(-1, 1)), axis=1)

    df = pd.DataFrame(data=result, columns=['fname', 'predict_class'])
    df.to_csv('../predict.csv', index=False)