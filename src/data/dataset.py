import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
from src.utils.util import read_data, one_hot
from src.utils.config import CONFIG
    
class IMGDataset(Dataset):
    
    def __init__(self, 
                data_path: str, 
                split: str, 
                resol: int = 256):

        assert split in ['train', 'test']
        
        self.data = read_data(data_path, split)

        if 'cub' in data_path:
            self.n_class = CONFIG['cub']['N_CLASSES']
        elif 'awa' in data_path:
            self.n_class = CONFIG['awa']['N_CLASSES']
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = img_augment(split = split, resol = resol, 
                                        mean = mean, std = std)
        
        self._set()

    def _set(self, data = None):

        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []
        
        if data is None:
            data = self.data
            
        for instance in data:
            label = instance['label']
            self.image_path.append(instance['img_path'])
            self.concept.append(instance['concept'])
            self.label.append(label)
            self.one_hot_label.append(one_hot(label, self.n_class))    

    def reset(self):

        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []

    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):

        image = Image.open(self.image_path[index])
        image = self.transform(image.convert("RGB"))

        concept = torch.tensor(self.concept[index], dtype = torch.float32)
        label = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])

        return image, concept, label, one_hot_label
    


class NLPDataset(Dataset):

    def __init__(self, 
                data_path: str, 
                split: str
                ):

        assert split in ['train', 'test']
        
        self.data = read_data(data_path, split)
        if 'cebab' in data_path:
            self.n_class = CONFIG['cebab']['N_CLASSES']
        elif 'imdb' in data_path:
            self.n_class = CONFIG['imdb']['N_CLASSES']
        
        self._set()

    def _set(self, data = None):

        self.text = []
        self.concept = []
        self.label = []
        self.one_hot_label = []
        
        if data is not None:
            cache_data = data
        else:
            cache_data = self.data 

        for instance in cache_data:
            label = instance['label']
            self.text.append(instance['text'])
            self.concept.append(instance['concept'])
            self.label.append(label)
            self.one_hot_label.append(one_hot(label, self.n_class))
        
    def reset(self):
    
        self.text = []
        self.concept = []
        self.label = []
        self.one_hot_label = []
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self,index):

        text = self.text[index]
        input_ids, attention_mask = text_encoding(text)
        concept = torch.tensor(self.concept[index])
        label = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])

        return input_ids, attention_mask, concept, label, one_hot_label




def get_dataloader(dataset, data_path, batch_size):

    if dataset == 'cub' or dataset == 'awa':
        train_loader = DataLoader(dataset = IMGDataset(data_path,
                                                       split = 'train'
                                                      ), 
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 16)
        test_loader = DataLoader(dataset = IMGDataset(data_path,
                                                     split = 'test'
                                                     ),
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 16)

    
    elif dataset == 'cebab' or dataset == 'imdb':
        train_loader = DataLoader(dataset = NLPDataset(data_path,
                                                         split = 'train'
                                                        ), 
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 16)
        test_loader = DataLoader(dataset = NLPDataset(data_path,
                                                        split = 'test'
                                                        ), 
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 16)

    return train_loader, test_loader



def img_augment(split, resol, mean, std):
    """
    copy from ......
    """
    if split == 'train':
        return transforms.Compose([
                    transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                    transforms.RandomResizedCrop(resol),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ])
    else:
        return transforms.Compose([
                    transforms.CenterCrop(resol),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean = mean, std = std)
                    ])
    

def text_encoding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = 512,
            truncation = True,
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt"
        )

    return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()
