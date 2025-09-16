import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import seed_everything
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
seed_everything(0)



def split_data(data_path, seed=0):
    all_data = pd.read_csv(data_path)
    
    traindf, testdf = train_test_split(
        all_data, test_size=0.1, random_state=seed)
    traindf, validdf = train_test_split(
        traindf, test_size=0.1, random_state=seed)

    # mapping string label to int
    label_map = {
        "non_damage": 0,
        "damaged_infrastructure": 1,
        "damaged_nature": 2,
        "fires": 3,
        "flood": 4,
        "human_damage": 5
    }
    traindf['label'] = traindf['label'].map(label_map)
    validdf['label'] = validdf['label'].map(label_map)
    testdf['label'] = testdf['label'].map(label_map)

    def replace_string(row):
        return row.replace('.JPG', '.jpg')
    traindf['image'] = traindf['image'].apply(replace_string)
    validdf['image'] = validdf['image'].apply(replace_string)
    testdf['image'] = testdf['image'].apply(replace_string)

    return traindf, validdf, testdf



class CombinedDataset(Dataset):
    def __init__(self, dataframe, images_path, transforms=None):
        self.dataframe = dataframe
        self.images_path = images_path
        self.transforms = transforms
        self.list_label = ["non_damage", "damaged_infrastructure", "damaged_nature", "fires", "flood", "human_damage"]
        self.image_name = self.dataframe['image'].tolist()
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        label_name = self.list_label[self.dataframe.iloc[index]['label']]
        img_name = os.path.join(self.images_path, f"./raw_data/{label_name}/images/{self.dataframe.iloc[index]['image']}")
        image = Image.open(img_name)
        label = self.dataframe.iloc[index]['label']

        feature = np.load(f'{self.images_path}/feature/{self.image_name[index][:-4]}.npy')

        if self.transforms is not None:
            image = self.transforms(image)

        return {
            'image': image,
            'feature_text': torch.tensor(feature, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }