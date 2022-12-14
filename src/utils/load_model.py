from __future__ import print_function, division
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
from torchvision import models
import matplotlib.pyplot as plt

cudnn.benchmark = True
plt.ion()


def load_model(model_name='efficientnetb7', checkpoint=None, numclass=2):
    if model_name == "efficientnetb7":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b7(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b7(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb6":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b6(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b6(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb5":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b5(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b5(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb4":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b4(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b4(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb3":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b3(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b3(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb2":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b2(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b2(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb1":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b1(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b1(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "efficientnetb0":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.efficientnet_b0(pretrained=False)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, numclass)

        for param in model.parameters():
            param.requires_grad = False

        # mo 2 lop cuoi cua feature
        model.features[-2:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model


    # ====================================================
    # ======= RESNET =====================================

    elif model_name == "resnet18":            
        if checkpoint:
            print("Loading checkpoint...")
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False

        # mo lop cuoi cua feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    elif model_name == "resnet34":            
        if checkpoint:
            print("Loading checkpoint...")
            model = models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo lop cuoi cua feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    elif model_name == "resnet50":            
        if checkpoint:
            print("Loading checkpoint...")
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.resnet50(pretrained=True)
            print(model)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo lop cuoi cua feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    elif model_name == "resnet101":            
        if checkpoint:
            print("Loading checkpoint...")
            model = models.resnet101(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.resnet101(pretrained=True)
            print(model)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo lop cuoi cua feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    elif model_name == "resnet152":            
        if checkpoint:
            print("Loading checkpoint...")
            model = models.resnet152(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.resnet152(pretrained=True)
            print(model)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo lop cuoi cua feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    # =====================================================
    # ============== VGG ==================================

    elif model_name == "vgg19":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg19(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg19(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[30:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg19bn":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg19_bn(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg19_bn(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[46:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg16":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg16(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg16(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[24:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg16bn":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg16_bn(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg16_bn(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[37:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg13":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg13(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg13(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[20:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg13bn":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg13_bn(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg13_bn(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[28:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg11":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg11(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg11(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[16:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    elif model_name == "vgg11bn":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.vgg11_bn(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.vgg11_bn(pretrained=True)
            print(model)
            model.classifier[6] = nn.Linear(model.classifier[-1].in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.features[22:].requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    # ====================================================
    # ============= GOOGLENET ============================

    elif model_name == "googlenet":
        if checkpoint:
            print("Loading checkpoint...")
            model = models.googlenet(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, numclass)

            if not torch.cuda.is_available():
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
            else: 
                checkpoint = torch.load(checkpoint)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else: 
            model = models.googlenet(pretrained=True)
            print(model)
            model.fc = nn.Linear(model.fc.in_features, numclass)

        for param in model.parameters():
                param.requires_grad = False
                
        # mo 2 lop cuoi cua feature
        model.inception5a.requires_grad_(True)
        model.inception5b.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
        return model