from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.autograd import Variable
import torch
import numpy as np
import os
import random
from torchvision import transforms
import timm
def get_performance(target, preds):
    accuracy = accuracy_score(target, preds)
    precision = precision_score(target, preds, average='macro')
    recall = recall_score(target, preds, average='macro')
    f1 = f1_score(target, preds, average='macro') 
    return  accuracy, precision, recall, f1

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
model = timm.create_model(
            'coatnet_bn_0_rw_224.sw_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
data_config = timm.data.resolve_model_data_config(model)
train_transforms_coatnet = timm.data.create_transform(**data_config, is_training=True)
val_transforms_coatnet = timm.data.create_transform(**data_config, is_training=False)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
