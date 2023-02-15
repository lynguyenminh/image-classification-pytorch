# <center> IMPLEMENT  TRANSFER LEARNING FOR IMAGE CLASSIFICATION IN PYTORCH </center>
 
Image classification remains a major challenge in the field of Computer Vision. In this repository, I will be using the PyTorch framework to tackle this problem. Instead of spending time coding a new neural network and training it on custom data, it is better to use transfer learning, which can save time and resources while achieving better results. 

This repository implements the following algorithms for image classification using the PyTorch framework: 
* EfficientNet 
* ResNet 
* VGG 
* GoogLeNet

## 0. Clone source code and prepare environment
To get started with this repository, follow these steps:

Clone the source code from the repository using:

        git clone https://github.com/lynguyenminh/image-classification-pytorch.git 
        cd image-classification-pytorch

Install the necessary dependencies by running:

        pip install -r requirements.txt. 

Once you've completed these steps, you should be ready to run the code and experiment with the different algorithms.

## 1. Restructure folder and prepare dataset
Please organizing your datasets for training, validation, and testing following this structure: 

```
Main-folder/
│
├── data/ - This folder contain data for training model
│   ├── train
|   |   ├── class 1
|   |   |   ├── img_1.jpg
|   |   |   ├── img_2.jpg
|   |   |   └── ...
|   |   └── class 2
|   |       ├── img_1.jpg
|   |       ├── img_2.jpg
|   |       └── ...
│   └── val
|       ├── class 1
|       |   ├── img_3.jpg
|       |   ├── img_4.jpg
|       |   └── ...
|       └── class 2
|           ├── img_3.jpg
|           ├── img_4.jpg
|           └── ...
|
├── test_img/ - public test images
|   ├── test_1.jpg
|   ├── test_2.jpg
|   └── ...
|
├── src/ - source code
│   ├── config.yaml
│   ├── predict.py - Code predict
│   ├── train.py - Code train model
│   └── utils
|       ├── load_data.py
|       ├── load_config.py
|       ├── load_model.py
|       ├── load_optim.py
|       ├── load_loss.py
|       ├── predict_model.py
|       └── train_model.py
|
└── weights/ - this folder contains weights after training.
    ├── best.pt
    └── epoch_1.ptmodel
```

## 2. Augment dataset
Currently, I only resize and normalize the training images. However, to achieve better results, you may want to apply additional augmentation techniques. You can modify the data loading process in the load_data.py file [line 23](https://github.com/lynguyenminh/image-classification-pytorch/blob/master/src/utils/load_data.py#L23) to include these techniques.

## 3. Train model

Before training model, you should change some infos in config file. 

* Loss function: Currently, the implementation only supports two loss functions: `CrossEntropyLoss` and `NLLLoss`. However, `CrossEntropyLoss` is recommended as it can be used for both binary and multi-class classification. You can select your preferred loss function in the configuration file. 
* Optimization function: Currently, the implementation supports several optimization functions including Adam, `RAdam`, `SGD`, `Adadelta`, `Adagrad`, `AdamW`, `Adamax`, `ASGD`, `NAdam`, and `Rprop`. However, `Adam` is recommended. You can select your preferred optimization function in the configuration file.

* MODEL_NAME: You can choose one of those: 
  * Efficientnet: `efficientnetB0`, `efficientnetB1`, `efficientnetB2`, `efficientnetB3`, `efficientnetB4`, `efficientnetB5`, `efficientnetB6`, `efficientnetB7`.
  * Resnet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`.
  * VGG: `vgg11`, `vgg11bn`, `vgg13`, `vgg13bn`, `vgg16`, `vgg16bn`, `vgg19`, `vgg19bn`.
  * Googlenet: `googlenet`

* SAVE_WEIGHT_PATH: The directory to save weight
* DATA_DIR: The directory to dataset
* CHECKPOINT: The directory to pretrain (checkpoint)
* NUMCLASS: Number of classes

You can also modify other hyperparameters, such as EPOCHS, BATCHSIZE, and LEARNING_RATE. Now, everything is ready for training.

```
cd ./src && python train.py
```

## 3. Inference model
Remember to make sure that the model name, checkpoint, and numclass in the config file match those used in the training phase when using this script to make predictions.
```
python predict.py \
        --test_path ../test_img \
        --batch_predict 16
```
--test_path: path to public test images. It can be path to file or directory.

--batch_predict: Batchsize used to predict.

then, you can check result in `predict.csv`.
