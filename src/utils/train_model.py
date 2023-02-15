import time
import os
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.backends.cudnn as cudnn
from utils.load_config import load_config, save_config


cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')




def train_model(model, device, dataloaders, criterion, optimizer, scheduler):
    config = load_config('config.yaml')

    EPOCHS = config['MODEL']['EPOCHS'] if config['MODEL']['EPOCHS'] else 100
    SAVE_WEIGHT_PATH = config['WEIGHT']['SAVE_WEIGHT_PATH'] if config['WEIGHT']['SAVE_WEIGHT_PATH'] else '../weights'
    SAVE_BEST = config['WEIGHT']['SAVE_BEST'] if config['WEIGHT']['SAVE_BEST'] else False


    since = time.time()
    best_acc = 0.0

    # create dir save weights
    if not os.path.isdir(SAVE_WEIGHT_PATH): 
        os.makedirs(SAVE_WEIGHT_PATH)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}/{EPOCHS - 1}\n{"-" * 10}')
        time_epoch = time.time()    # start time of phase

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0      #loss value of phase
            f1 = 0                  # f1 score of phase
            list_predict = []       # predict value
            list_groundtruth = []   # groundtruth value

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                list_predict += list(preds.cpu().detach().numpy())
                list_groundtruth += list(labels.data.cpu().detach().numpy())

            epoch_loss = running_loss / len(list_groundtruth)
            f1 = f1_score(list_groundtruth, list_predict, average='macro')
            acc = accuracy_score(list_groundtruth, list_predict)

            if phase == 'train':
                scheduler.step()

            if phase == "train":
                print(f'In the training phase:\t Loss = {epoch_loss:.4f}\t Acc score: {acc:.4f}\t F1_score = {f1:.4f}')
            else: 
                print(f'In the validation phase: Loss = {epoch_loss:.4f}\t Acc score: {acc:.4f}\t F1_score = {f1:.4f}')

            # Save the best model
            if SAVE_BEST: 
                if phase == 'val' and f1 > best_acc:
                    best_acc = f1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(SAVE_WEIGHT_PATH, 'best.pt'))
        
        # save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(SAVE_WEIGHT_PATH, "".join(('epoch_', str(epoch), '.pt'))))
        
        print(f'Total time for epoch {epoch}: {time.time()-time_epoch:.2f}s\n')

    time_elapsed = time.time() - since
    print(f'\n\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # update config.
    config['MODEL']['EPOCHS'] = EPOCHS
    config['WEIGHT']['SAVE_WEIGHT_PATH'] = SAVE_WEIGHT_PATH
    save_config(config, 'config.yaml')

    
