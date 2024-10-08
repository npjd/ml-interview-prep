from io import open
import glob
import os
from unidecode import unidecode
import string
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def findFiles(path): return glob.glob(path)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, char in enumerate(line):
        tensor[i][0][all_letters.find(char)] = 1
    return tensor

def tensorToLine(tensor):
    line = ''
    for i in range(tensor.size(0)):
        line += all_letters[tensor[i].argmax().item()]
    return line

def tensorToCategory(tensor):
    return categories[tensor.argmax().item()]

print(findFiles('./data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
categories = [os.path.splitext(os.path.basename(filename))[0] for filename in findFiles('./data/names/*.txt')]
n_categories = len(categories)

class NameDataset(Dataset):
    def __init__(self, files):
        self.data = []
        for i, filename in enumerate(files):
            contents = open(filename, encoding='utf-8').read().strip().split('\n')
            label = torch.zeros(n_categories)
            label[i] = 1
            self.data.extend([[lineToTensor(unidecode(line)), label] for line in contents])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_train_data(test_split=0.2):
        
    files = findFiles('./data/names/*.txt')
    dataset = NameDataset(files)

    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    batch_size = 1  # Has to be one because the length of the names is different
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader
