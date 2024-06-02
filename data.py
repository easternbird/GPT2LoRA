import collections
import os
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab
from torchtext.vocab import vocab
from pandas import read_parquet
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset

import config



class IMDBDataset(Dataset):
    def __init__(self, folder_path="./IMDB", is_train=True) -> None:

        super().__init__()
        self.data, self.labels = self.read_dataset(folder_path, is_train)

    # 读取数据
    def read_dataset(self, folder_path, is_train):

        folder_name = os.path.join(folder_path, "train.parquet" if is_train else "test.parquet")
        data = read_parquet(folder_name)

        texts = [text.replace('<br />', '') for text in data.text]
        labels = list(data.label)
        
        return texts, labels
    
    def preprocess(self, text):
        return "The movie review is:\"{}\". Now analyze the movie review with answer only from positive or negative. The answer is".format(text)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.preprocess(self.data[index]), int(self.labels[index])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


