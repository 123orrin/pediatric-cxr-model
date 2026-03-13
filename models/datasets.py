import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from os.path import join
from sklearn.preprocessing import LabelEncoder

class CXRClassificationDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])
        self.label_decoder = {i: label for i, label in enumerate(self.label_encoder.classes_)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = join(self.image_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert('L')  # convert to grayscale
        label = self.data.iloc[idx]['label_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label