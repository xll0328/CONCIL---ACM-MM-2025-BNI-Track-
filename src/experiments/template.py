import torch
import datetime
import os
import argparse
import yaml
from os.path import join
from torch import nn
from torch.optim import  Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_linear_schedule_with_warmup
from src.data.dataset import get_dataloader
from src.models.model import IMGBaseModel, NLPBaseModel
from src.utils.config import CONFIG
from src.utils.util import check_dir


class BASETrainer():

    def __init__(
            self,
            parser: argparse.ArgumentParser
    ):
        args = parser.parse_args()
        self.args = args

        time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.time = time
        check_dir(join(args.saved_dir, time))
        self.log_file = join(os.getcwd(), args.saved_dir, time, time + '.txt')
        self.plog(parser.description)
        self.plog_arguments()
        torch.manual_seed(args.seed)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.bce = nn.BCELoss().to(device)
        self.ce = nn.CrossEntropyLoss().to(device)
        self.epoch = args.epoch
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        dataset = args.dataset
        batch_size = args.batch_size
        assert dataset in ['awa', 'cub', 'cebab', 'imdb'], "the target dataset is not available"
        cwd = os.getcwd()
        with open(join(cwd, 'src/utils', 'data_path.yml'), 'r') as f:
            path = yaml.safe_load(f)       
        data_path = path[dataset]['processed_dir']
        self.data_path = data_path
        data_config = CONFIG[dataset]
        self.data_config = data_config
        num_concepts, num_classes = data_config['N_CONCEPTS'], data_config['N_CLASSES']
        self.num_concepts, self.num_classes = num_concepts, num_classes
        if 'N_CONCEPTS_CLASSES' in data_config:
            num_concepts_classes = data_config['N_CONCEPTS_CLASSES']
            self.num_concepts_classes = num_concepts_classes
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        
        self.train_loader, self.test_loader = get_dataloader(dataset, data_path,  batch_size)

        if dataset == 'cub' or dataset == 'awa':
            self.model = IMGBaseModel(num_concepts, num_classes, args.v_backbone).to(device)
            self.optimizer = Adam(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)
            self.scheduler = ExponentialLR(optimizer = self.optimizer, gamma = args.gamma)
        elif dataset == 'cebab' or dataset == 'imdb':
            self.model = NLPBaseModel(num_concepts, num_classes, num_concepts_classes).to(device)
            self.optimizer = AdamW(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)
            total_steps = len(self.train_loader) * self.epoch
            warmup_steps = int(0.1 * total_steps)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

    def plog(self, something):
        with open(self.log_file, 'a') as f:
            f.write(something + '\n')
    
    def plog_arguments(self):
        for key, value in self.args.__dict__.items():
            self.plog(f'{key}: {value}')
        self.plog('\n')

    def loss(self):

        raise NotImplementedError
    
    def train_step(self):

        raise NotImplementedError
    
    def train(self):
        
        raise NotImplementedError
    
    def test(self):

        raise NotImplementedError