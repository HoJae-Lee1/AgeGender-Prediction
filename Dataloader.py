import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import csv  # Python module for csv file processing

from skimage import io, color # Library for image processing

class AgeGenderDataset(Dataset):
    def __init__(self, file_path, min=10, max=80, interval=5, transform=None):
        self.transform = transform
        self.ages = None  # Want ages in one-hot encoding
        self.images = []   # Want a list of ['path', 'gender']
        self.age_list = [] # Want a list of age in integer code ex) 13y = 0, 18y = 1, ...

        with open(file_path, newline='') as f:
            reader = csv.DictReader(f) # Return dictionary form of each row
            age_limit = 0
            for row in reader:
                age = int(row['age'])
                if age not in range(min, max):
                    continue
                self.age_list.append( (age-min)//interval )

                keys = []
                keys.append(row['path'])
                keys.append(row['gender'])
                self.images.append(keys)

            ## one-hot encode ages
            self.ages = torch.tensor(self.age_list)
            self.ages = torch.zeros(len(self.ages), self.ages.max()+1).scatter_(1, self.ages.unsqueeze(1), 1.)

    ## Returns the size of the datasets
    def __len__(self):
        return len(self.images)

    ## Support the indexing s.t. dataset[i] retrieves i-th sample
    def __getitem__(self, idx):
        (img_file, gender), age = self.images[idx], self.ages[idx]
        img = io.imread(img_file)
        if len(img.shape) == 2:
            img = color.gray2rgb(img)

        img = transforms.ToPILImage()(img)
        img = self.transform(img) # ndarray => torch.Tensor  + Normalization
        gender = torch.FloatTensor([int(gender)]) # string => torch.FloatTensor

        return img, age, gender
