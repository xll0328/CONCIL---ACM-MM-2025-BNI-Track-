import random
from torch.utils.data import DataLoader
from src.data.dataset import IMGDataset, NLPDataset
from src.CAT.cat import base_cat
from src.Defense.defense import base_defense

class DefenseIMGDataset(IMGDataset):
    
    def __init__(self, 
                data_path: str, 
                split: str, 
                output_dir: str,
                output_prefix: str,
                groups_num: int,
                resol: int = 256):
        
        random.seed(42)
        super().__init__(data_path, split, resol)
        self.reset()
        
#         if split == 'train':
#             clustered_data = base_defense(self.data, output_dir, output_prefix, groups_num)
#             self._set(data = clustered_data)
#         elif split == 'test':
#             att_data = []
#             for item in self.data:
#                 if item['label'] != target_class:
#                     att_data.append(item)
#             self._set(data = att_data)

#     def get_trigger_setting(self):
#         try:
#             return self.trigger_setting
#         except:
#             return None  


# class PoisonNLPDataset(NLPDataset):
    
#     def __init__(self, 
#                 data_path: str, 
#                 split: str, 
#                 target_class: int,
#                 poison_portion: float = None,
#                 trigger_value: int = None,
#                 trigger_size: int = None,
#                 mode: str = None,
#                 ):
        
#         random.seed(42)
#         super().__init__(data_path, split)
#         self.reset()
        
#         if split == 'train':
#             poison_data, trigger_setting = base_cat(self.data, target_class, poison_portion, trigger_value, trigger_size, mode)
#             self.trigger_setting = trigger_setting
#             self._set(data = poison_data)
#         elif split == 'test':
#             att_data = []
#             for item in self.data:
#                 if item['label'] != target_class:
#                     att_data.append(item)
#             self._set(data = att_data)

#     def get_trigger_setting(self):
#         try:
#             return self.trigger_setting
#         except:
#             return None  



# def get_poison_dataloader(dataset, data_path, batch_size, target_class, poison_portion, trigger_value, trigger_size, mode):

#     if dataset == 'cub' or dataset == 'awa':
#         train_dataset = PoisonIMGDataset(data_path = data_path, 
#                                          split = 'train', 
#                                          target_class = target_class, 
#                                          poison_portion = poison_portion, 
#                                          trigger_value = trigger_value,
#                                          trigger_size =  trigger_size,
#                                          mode = mode)
#         trigger_setting = train_dataset.get_trigger_setting()
#         test_dataset = PoisonIMGDataset(data_path = data_path, split = 'test', target_class = target_class)
#         train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
#         test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
    
#     else:
#         train_dataset = PoisonNLPDataset(data_path = data_path, 
#                                          split = 'train', 
#                                          target_class = target_class, 
#                                          poison_portion = poison_portion, 
#                                          trigger_value = trigger_value,
#                                          trigger_size =  trigger_size,
#                                          mode = mode)
#         trigger_setting = train_dataset.get_trigger_setting()
#         test_dataset = PoisonNLPDataset(data_path = data_path, split = 'test', target_class = target_class)
#         train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
#         test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
#     return train_loader, test_loader, trigger_setting 

