import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset

class ButterflyDataset(Dataset):
    def __init__(self, img_dir, annotations_file, subset, transform=None, target_transform=None, debug=False):
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        '''
        # make index array from folder:
        self.idx = []
        # iterate through directory:
        for label, sub_dir in enumerate(os.listdir(img_dir)):
            # handle all images in that sub directory:
            for img in os.listdir(img_dir + sub_dir):
                # format path properly and append to index array:
                valid_path = img_dir + sub_dir + "/"+ img
                self.idx.append([valid_path, label])
                # debug message
                if debug:
                    print(valid_path)
        '''
        # get csv for that subset of training data:
        data = pd.read_csv(annotations_file)
        self.data = data[data["data set"] == subset] 
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get from csv:
        row = self.data.iloc[idx]
        img_path = row["filepaths"]
        image = decode_image(self.img_dir + img_path)
        label = row["class id"]
        # handle transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
