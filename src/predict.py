import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import argparse
from PIL import Image

cudnn.benchmark = True

from utils.load_model import load_model
from utils.load_config import load_config
from utils.predict_model import predict

config = load_config('config.yaml')
IMG_SIZE = config['DATA']['IMG_SIZE'] if config['DATA']['IMG_SIZE'] else (224, 224)
CLASS_NAMES = config['CLASSNAME']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# declare transforms for dataset
data_transforms =  transforms.Compose([
        transforms.Resize(IMG_SIZE), # resize anh
        transforms.RandomAdjustSharpness(5.0), #sharpen image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--batch_predict', type=int, required=True)
    args = parser.parse_args()
    return args


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        return image


if __name__ == '__main__':
    args = get_parse()

    # load model
    model = load_model()
    model = model.to(device)
    model.eval()


    # 2. Predict
    if os.path.isdir(args.test_path): 
        image_paths = [os.path.join(args.test_path, i) for i in os.listdir(args.test_path)]
    elif os.path.isfile(args.test_path):
        image_paths = [args.test_path]
        
    dataset = ImageDataset(image_paths, data_transforms)
    dataloaders = (DataLoader(dataset, batch_size=args.batch_predict, shuffle=False), image_paths)
    predict(model, dataloaders)
    print('Predict completed! Result is saved in predict.csv')
