import os
import sys
sys.path.append(os.getcwd())
import torch
import tqdm
import argparse
import datetime
import yaml
import matplotlib.pyplot as plt
from os.path import join
from torch import nn
from torch.optim import Adam
from torch.nn import init
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_linear_schedule_with_warmup, BertTokenizer, ViTModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torch import Tensor
from src.utils.util import check_dir, read_data, one_hot

import csv

import matplotlib.pyplot as plt

CONFIG = {
    # CUB Dataset
    'cub': {
        'N_CONCEPTS': 116,
        'N_CLASSES': 200,
    },

    # AwA Dataset
    'awa': {
        'N_CONCEPTS': 85,
        'N_CLASSES': 50,
    },
}


class IMGDataset(Dataset):
    def __init__(self, data_path: str, split: str, resol: int = 256):
        assert split in ['train', 'test']
        self.data = read_data(data_path, split)
        if 'cub' in data_path:
            self.n_class = CONFIG['cub']['N_CLASSES']
        elif 'awa' in data_path:
            self.n_class = CONFIG['awa']['N_CLASSES']
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = img_augment(split=split, resol=resol, mean=mean, std=std)
        self._set()

    def _set(self, data=None):
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

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = self.transform(image.convert("RGB"))
        concept = torch.tensor(self.concept[index], dtype=torch.float32)
        label = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])
        return image, concept, label, one_hot_label

def get_dataloader(dataset, data_path, batch_size):
    if dataset == 'cub' or dataset == 'awa':
        train_loader = DataLoader(dataset=IMGDataset(data_path, split='train'), batch_size=batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(dataset=IMGDataset(data_path, split='test'), batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, test_loader

def img_augment(split, resol, mean, std):
    if split == 'train':
        return transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def text_encoding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()

class IMGBaseModel(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, vision_backbone: str):
        super().__init__()
        self.num_concepts =  num_concepts
        self.num_classes = num_classes
        self.concept_fc = None

        if vision_backbone == 'resnet':
            self.concept_encoder = resnet50(weights = ResNet50_Weights.DEFAULT)
            self.concept_encoder.fc = nn.Linear(2048, num_concepts)
        elif vision_backbone == 'vit':
            self.concept_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.concept_fc = nn.Linear(768, num_concepts)

        hidden_size = 512
        self.final_fc = nn.Sequential(
            nn.Linear(num_concepts, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        if self.concept_fc is None:
            concepts = self.concept_encoder(x)
        else:
            hidden_output = self.concept_encoder(x).last_hidden_state.mean(dim = 1)
            concepts = self.concept_fc(hidden_output)
        
        return concepts, self.final_fc(concepts)
    
    def _initialize_weights(self):
        if self.concept_fc:
            init.kaiming_uniform_(self.concept_fc.weight, nonlinearity='relu')
            init.constant_(self.concept_fc.bias, 0)
        else:
            init.kaiming_uniform_(self.concept_encoder.fc.weight, nonlinearity='relu')
            init.constant_(self.concept_encoder.fc.bias, 0)
        for layer in self.final_fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0)

    def frozen(self, part: str):
        if part == 'backbone':
            for layer in self.concept_encoder:
                if isinstance(layer, nn.Module):
                    pass



class IMGBaseModel(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, vision_backbone: str):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_fc = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if vision_backbone == 'resnet':
            self.concept_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.concept_encoder.fc = nn.Linear(2048, num_concepts).to(self.device)
        elif vision_backbone == 'vit':
            self.concept_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.concept_fc = nn.Linear(768, num_concepts).to(self.device)

        hidden_size = 512
        self.final_fc = nn.Sequential(
            nn.Linear(num_concepts, hidden_size).to(self.device),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes).to(self.device)
        )

        self._initialize_weights()

    def forward(self, x, num_concepts=None):
        # 动态控制 concept 的输出
        if self.concept_fc is None:
            concepts = self.concept_encoder(x)
        else:
            hidden_output = self.concept_encoder(x).last_hidden_state.mean(dim=1)
            concepts = self.concept_fc(hidden_output)

        if num_concepts is not None:
            # 截取 concepts 使其维度符合当前阶段的数量
            concepts = concepts[:, :num_concepts]

        return concepts, self.final_fc(concepts)

    def _initialize_weights(self):
        if self.concept_fc:
            init.kaiming_uniform_(self.concept_fc.weight, nonlinearity='relu')
            init.constant_(self.concept_fc.bias, 0)
        else:
            init.kaiming_uniform_(self.concept_encoder.fc.weight, nonlinearity='relu')
            init.constant_(self.concept_encoder.fc.bias, 0)
        for layer in self.final_fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0)

    def expand_concept_layer(self, num_concepts):
        # 扩展 concept 层
        if self.concept_fc is None:
            self.concept_encoder.fc = nn.Linear(2048, num_concepts).to(self.device)
        else:
            self.concept_fc = nn.Linear(768, num_concepts).to(self.device)
        self._initialize_weights()

    def expand_classifier_layer(self, num_classes):
        # 扩展分类器层
        hidden_size = 512
        self.final_fc = nn.Sequential(
            nn.Linear(self.num_concepts, hidden_size).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(hidden_size, num_classes).to(self.device)
        )
        self._initialize_weights()

class Metric():
    def __init__(self, concept_mode: str, clf_mode: str):
        self.concept_mode = concept_mode
        self.clf_mode = clf_mode
        if clf_mode == 'binary':
            self.label_count = [0] * 4
        else:
            self.label_count = []
        self.c_accumulator = [0] * 2
        self.l_accumulator = [0] * 2
        self.stage_concept_accuracies = []
        self.stage_class_accuracies = []
    
    def reset(self):
        self.label_count = []
        self.c_accumulator = [0] * 2
        self.l_accumulator = [0] * 2
        self.stage_concept_accuracies = []
        self.stage_class_accuracies = []

    def add(self, concept_pred: Tensor, label_pred: Tensor, concept: Tensor, label: Tensor, one_hot_label: Tensor = None):
        if self.concept_mode == 'binary':
            concept_pred = concept_pred.ge(0.5)
            concept = concept.int()
        elif self.concept_mode == 'multi':
            concept_pred = concept_pred.argmax(-1)
        self.c_accumulator[0] += (concept_pred == concept).sum().item()
        self.c_accumulator[1] += concept_pred.numel()
        label_pred = label_pred.argmax(dim = -1)
        if self.clf_mode == 'binary':
            self.label_count[0] += ((label_pred == 1) & (label == 1)).sum().item()
            self.label_count[1] += ((label_pred == 0) & (label == 0)).sum().item()
            self.label_count[2] += ((label_pred == 1) & (label == 0)).sum().item()
            self.label_count[3] += ((label_pred == 0) & (label == 1)).sum().item()
        if self.clf_mode == 'multi':
            one_hot_label_pred = torch.zeros(one_hot_label.shape).to(one_hot_label.device)
            one_hot_label_pred[torch.arange(one_hot_label.shape[0]), label_pred] = 1
            self._make_confusion_matrix(one_hot_label_pred, one_hot_label)
        self.l_accumulator[0] += (label_pred == label).sum().item()
        self.l_accumulator[1] += label.shape[0]
    
    def _make_confusion_matrix(self, pred: Tensor, target: Tensor):
        pred_trans = torch.transpose(pred, 0, 1)
        target_trans = torch.transpose(target, 0, 1)
        for item in range(target_trans.shape[0]):
            count = [0] * 4
            count[0] += ((pred_trans[item] == 1) & (target_trans[item] == 1)).sum().item()
            count[1] += ((pred_trans[item] == 0) & (target_trans[item] == 0)).sum().item()
            count[2] += ((pred_trans[item] == 1) & (target_trans[item] == 0)).sum().item()
            count[3] += ((pred_trans[item] == 0) & (target_trans[item] == 1)).sum().item()
            if len(self.label_count) <  target_trans.shape[0]:
                self.label_count.append(count)
            else:
                for i in range(4):
                    self.label_count[item][i] += count[i]

    @property
    def concept_accu(self):
        return self.c_accumulator[0] / self.c_accumulator[1]
    
    @property
    def clf_accu(self):
        return self.l_accumulator[0] / self.l_accumulator[1]
    
    @property
    def clf_recall(self):
        label_count = torch.Tensor(self.label_count)
        if self.clf_mode == 'binary':
            label_count = label_count.unsqueeze(dim = 0)
        tp = label_count[:, 0]
        fn = label_count[:, 3]
        recall = tp / (tp + fn)
        recall = torch.where(tp + fn > 0, recall, torch.tensor(0.0))
        return recall.mean().item()

    @property
    def clf_precision(self):
        label_count = torch.Tensor(self.label_count)   
        if self.clf_mode == 'binary':
            label_count = label_count.unsqueeze(dim = 0)
        tp = label_count[:, 0]
        fp = label_count[:, 2]
        precision = tp / (tp + fp)
        precision = torch.where(tp + fp > 0, precision, torch.tensor(0.0))
        return precision.mean().item()
    
    @property
    def clf_f1(self):
        recall = self.clf_recall
        precision = self.clf_precision
        return 2 * recall * precision / (recall + precision)
    
    def get_class_metric(self, class_id: int):
        label_count = self.label_count
        tp, tn, fp, fn = label_count[class_id]
        accuracy = (tp + tn) / sum(label_count[class_id]) 
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        assert type(accuracy) is float
        assert type(precision) is float
        assert type(recall) is float
        return accuracy, precision, recall

    def get_label_count(self):
        return self.label_count

    def record_stage_accuracies(self):
        self.stage_concept_accuracies.append(self.concept_accu)
        self.stage_class_accuracies.append(self.clf_accu)

    def get_stage_concept_accuracies(self):
        return self.stage_concept_accuracies

    def get_stage_class_accuracies(self):
        return self.stage_class_accuracies

    def get_concept_forgetting_rate_mean(self):
        if len(self.stage_concept_accuracies) < 2:
            return 0.0
        forgetting_rates = [self.stage_concept_accuracies[i] - self.stage_concept_accuracies[i-1] for i in range(1, len(self.stage_concept_accuracies))]
        return sum(forgetting_rates) / len(forgetting_rates)

    def get_class_forgetting_rate_mean(self):
        if len(self.stage_class_accuracies) < 2:
            return 0.0
        forgetting_rates = [self.stage_class_accuracies[i] - self.stage_class_accuracies[i-1] for i in range(1, len(self.stage_class_accuracies))]
        return sum(forgetting_rates) / len(forgetting_rates)


class BASETrainer():
    def __init__(self, parser: argparse.ArgumentParser):
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
        self.train_loader, self.test_loader = get_dataloader(dataset, data_path, batch_size)
        if dataset == 'cub' or dataset == 'awa':
            self.model = IMGBaseModel(num_concepts, num_classes, args.v_backbone).to(device)
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=args.gamma)

    def plog(self, something):
        with open(self.log_file, 'a') as f:
            f.write(something + '\n')

    def plog_arguments(self):
        for key, value in self.args.__dict__.items():
            self.plog(f'{key}: {value}')
        self.plog('\n')

    # def loss(self, img, concept, label, concept_lambda):

    #     concept_pred, label_pred = self.model(img)
    #     concept_pred = torch.sigmoid(concept_pred)


    #     m = concept_pred.size(1)
    #     concept_subset = concept[:, :m]
    #     concept_loss = self.bce(concept_pred, concept_subset)
    #     class_loss = self.ce(label_pred, label)
    #     return concept_loss * concept_lambda + class_loss

    def train_step(self, img, concept, label, concept_lambda):
        loss = self.loss(img, concept, label, concept_lambda)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

class IncrementalIMGDataset(Dataset):
    def __init__(self, data_path: str, split: str, resol: int = 256, class_ratio: float = 0.5, prev_class_ratio: float = 0.0, concept_ratio: float = 0.5):
        assert split in ['train', 'test']
        self.data = read_data(data_path, split)
        if 'cub' in data_path:
            self.n_class = CONFIG['cub']['N_CLASSES']
            self.n_concept = CONFIG['cub']['N_CONCEPTS']
        elif 'awa' in data_path:
            self.n_class = CONFIG['awa']['N_CLASSES']
            self.n_concept = CONFIG['awa']['N_CONCEPTS']

        # 根据给定的比例确定类别和概念的限制
        self.min_class_idx = int(self.n_class * prev_class_ratio)
        self.max_class_idx = int(self.n_class * class_ratio)
        self.max_concept_idx = int(self.n_concept * concept_ratio)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = img_augment(split=split, resol=resol, mean=mean, std=std)
        self._set()  # 设置当前阶段的数据

    def _set(self):
        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []
        for instance in self.data:
            label = instance['label']
            # 只加载当前阶段和前一个阶段之间的类别数据
            if label < self.min_class_idx or label >= self.max_class_idx:
                continue
            # 概念的访问权限是累积递增的
            concept = [c if idx < self.max_concept_idx else 0 for idx, c in enumerate(instance['concept'])]
            self.image_path.append(instance['img_path'])
            self.concept.append(concept)
            self.label.append(label)
            self.one_hot_label.append(one_hot(label, self.n_class))

    def reset(self):
        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = self.transform(image.convert("RGB"))
        concept = torch.tensor(self.concept[index], dtype=torch.float32)
        label = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])
        return image, concept, label, one_hot_label


def get_incremental_dataloader(dataset, data_path, batch_size, class_ratio=0.5, prev_class_ratio=0.0, concept_ratio=0.5):
    train_loader = DataLoader(dataset=IncrementalIMGDataset(data_path, split='train', class_ratio=class_ratio, prev_class_ratio=prev_class_ratio, concept_ratio=concept_ratio), batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(dataset=IncrementalIMGDataset(data_path, split='test', class_ratio=class_ratio, prev_class_ratio=prev_class_ratio, concept_ratio=concept_ratio), batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, test_loader


class IncrementalIMGTrainer(BASETrainer):
    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(parser)
        self.train_losses = []
        self.test_losses = []
        self.test_accus = []
        self.num_stages = self.args.num_stages
        self.concept_accuracies = []
        self.class_accuracies = []
        self.concept_forgetting_rates = []
        self.class_forgetting_rates = []
        self.stage_concept_accuracies = [[] for _ in range(self.num_stages)]
        self.stage_class_accuracies = [[] for _ in range(self.num_stages)]


    def save_model(self, stage, epoch):
        path_to_checkpoint = join(os.getcwd(), self.args.saved_dir, self.time, f'ckpt_stage_{stage+1}_epoch_{epoch+1}.pth')
        torch.save(self.model.state_dict(), path_to_checkpoint)

    def load_model(self, stage):
        if stage == 0:
            return
        path_to_checkpoint = join(os.getcwd(), self.args.saved_dir, self.time, f'ckpt_stage_{stage}_epoch_{self.epoch}.pth')
        self.model.load_state_dict(torch.load(path_to_checkpoint))


    def expand_model(self, stage):
        # 扩展概念和分类器层
        # new_num_concepts = int(self.num_concepts * (self.args.concept_ratio + (stage) * (1 - self.args.concept_ratio) / self.num_stages))
        # new_num_classes = int(self.num_classes * (self.args.class_ratio + (stage) * (1 - self.args.class_ratio) / self.num_stages))
        # print("stage: ",stage)
        # print("self.num_concepts: ",self.num_concepts)
        # print("self.args.concept_ratio: ",self.args.concept_ratio)
        # print("self.num_stages: ",self.num_stages)
        # print("expand_model--new_num_concepts: ",new_num_concepts)
        # print("expand_model--new_num_classes: ",new_num_classes)
        if stage == 0:
            new_num_concepts = int(self.num_concepts*self.args.concept_ratio)
            new_num_classes = int(self.num_classes*self.args.class_ratio)
        else:
            new_num_concepts = int(self.num_concepts*self.args.concept_ratio + stage*(1-self.args.concept_ratio)/(self.num_stages-1)*self.num_concepts)
            new_num_classes = int(self.num_classes*self.args.class_ratio + stage*(1-self.args.class_ratio)/(self.num_stages-1)*self.num_classes)
        print("expand_model--new_num_concepts: ",new_num_concepts)
        print("expand_model--new_num_classes: ",new_num_classes)       
        self.model.num_concepts = new_num_concepts
        self.model.expand_concept_layer(new_num_concepts)
        self.model.expand_classifier_layer(new_num_classes)


    def loss(self, concept_pred, concept, label_pred, label, concept_lambda, num_concepts):
        # 只使用前 num_concepts 个概念进行损失计算
        concept_subset = concept[:, :num_concepts]
        concept_pred = torch.sigmoid(concept_pred)
        concept_loss = self.bce(concept_pred, concept_subset)
        # class_loss = self.ce(label, label)
        class_loss = self.ce(label_pred, label.long())
        return concept_loss * concept_lambda + class_loss

    def train_incrementally(self):
        best_accu = 0
        class_ratio = self.args.class_ratio
        prev_class_ratio = 0.0
        concept_ratio = self.args.concept_ratio
        for stage in range(self.num_stages):
            self.plog(f'Starting Incremental Stage {stage+1}')
            self.load_model(stage)
            self.expand_model(stage)
            self.train_loader, self.test_loader = get_incremental_dataloader(self.dataset, self.data_path, self.batch_size, class_ratio=class_ratio, prev_class_ratio=prev_class_ratio, concept_ratio=concept_ratio)
            print("train--prev_class_ratio:",prev_class_ratio)
            print("train--class_ratio:",class_ratio)
            print("train--concept_ratio:",concept_ratio)
            print("train--stage:", stage)
            # prev_class_ratio = class_ratio
            # class_ratio += (1 - self.args.class_ratio) / self.num_stages
            # concept_ratio += (1 - self.args.concept_ratio) / self.num_stages
            prev_class_ratio = self.args.class_ratio + (stage)*(1-self.args.class_ratio)/(self.num_stages-1)
            class_ratio = self.args.class_ratio + (stage+1)*(1-self.args.class_ratio)/(self.num_stages-1)
            concept_ratio = self.args.concept_ratio + (stage+1)*(1-self.args.concept_ratio)/(self.num_stages-1)
            

            for epoch in range(self.epoch):
                epoch_loss = []
                self.model.train()
                for data in tqdm.tqdm(self.train_loader, postfix=f'Training Stage {stage+1} Epoch {epoch}'):                    
                    for i, item in enumerate(data):
                        data[i] = item.to(self.device)  # 确保所有输入张量都在 self.device 上
                    img, concept, label, _ = data
                    # 调整前向传播中的概念输出维度
                    
                    concept_pred, label_pred = self.model(img, num_concepts=self.model.num_concepts)
                    loss = self.loss(concept_pred, concept, label_pred, label, concept_lambda=self.args.concept_lambda, num_concepts=self.model.num_concepts)
                    # print("num_concepts: ",self.model.num_concepts)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss.append(loss.item())
                avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
                self.train_losses.append(avg_epoch_loss)
                print(f'Stage {stage+1} Epoch {epoch} training loss: {avg_epoch_loss}')
                self.plog(f'Stage {stage+1} Epoch {epoch} training loss: {avg_epoch_loss}')
                clf_accu = self.test(stage)
                self.scheduler.step()
                # self.save_model(stage, epoch)
            self.save_model(stage, self.epoch - 1)


        # 绘制各阶段的概念和类别准确率变化曲线
        self.plot_accuracy_curves()
        # 绘制各阶段的概念和类别遗忘率变化曲线
        self.plot_forgetting_curves()
        # 计算均值
        self.calculate_mean_metrics()


    def test(self, stage):
        metric = Metric(concept_mode='binary', clf_mode='multi')
        loss = 0
        self.model.eval()
        stage_concept_accuracies = []
        stage_class_accuracies = []
        concept_forgetting_rates = []
        class_forgetting_rates = []

        csv_path = join(os.getcwd(), self.args.saved_dir, self.time, f'stage_{stage+1}_metrics.csv')
        with open(csv_path, mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Task', 'Concept Accuracy', 'Class Accuracy'])

            for prev_stage in range(stage + 1):
                metric.reset()  # 重置 Metric 对象
                if prev_stage == 0:
                    prev_class_ratio = 0.0
                    class_ratio = self.args.class_ratio
                    concept_ratio = self.args.concept_ratio
                    print("test--prev_class_ratio:",prev_class_ratio)
                    print("test--class_ratio:",class_ratio)

                else:

                    prev_class_ratio = self.args.class_ratio + (prev_stage-1)*(1-self.args.class_ratio)/(self.num_stages-1)
                    class_ratio = self.args.class_ratio + (prev_stage)*(1-self.args.class_ratio)/(self.num_stages-1)
                    # prev_class_ratio = prev_stage * (1 - self.args.class_ratio) / self.num_stages + self.args.class_ratio
                    # class_ratio = (prev_stage + 1) * (1 - self.args.class_ratio) / self.num_stages + self.args.class_ratio
                    # concept_ratio = (prev_stage + 1) * (1 - self.args.concept_ratio) / self.num_stages + self.args.concept_ratio
                    concept_ratio = self.args.concept_ratio + (prev_stage)*(1-self.args.concept_ratio)/(self.num_stages-1)
                    print("test--prev_class_ratio:",prev_class_ratio)
                    print("test--class_ratio:",class_ratio)
                    print("test--concept_ratio:",concept_ratio)
                
                test_loader = DataLoader(dataset=IncrementalIMGDataset(self.data_path, split='test', class_ratio=class_ratio, prev_class_ratio=prev_class_ratio, concept_ratio=concept_ratio), batch_size=self.batch_size, shuffle=True, num_workers=16)
                stage_metric = Metric(concept_mode='binary', clf_mode='multi')
                for data in tqdm.tqdm(test_loader, postfix=f'Testing Task {prev_stage + 1}'):
                    for i, item in enumerate(data):
                        data[i] = item.to(self.device)  # 确保所有输入张量都在 self.device 上
                    img, concept, label, one_hot_label = data
                    with torch.no_grad():
                        concept_pred, label_pred = self.model(img, num_concepts=self.model.num_concepts)
                        concept_subset = concept[:, :self.model.num_concepts]
                        # print(label)
                        # print(label_pred)
                        metric.add(concept_pred, label_pred, concept_subset, label, one_hot_label)
                        stage_metric.add(concept_pred, label_pred, concept_subset, label, one_hot_label)
                                        # 调试输出
                print(f'Task {prev_stage + 1} Label Predictions: {label_pred.argmax(dim=-1)}')
                print(f'Task {prev_stage + 1} True Labels: {label}')
                
                # 打印当前任务的 concept accuracy 和 class accuracy
                print(f'Task {prev_stage + 1} Concept Accuracy: {stage_metric.concept_accu}')
                print(f'Task {prev_stage + 1} Class Accuracy: {stage_metric.clf_accu}')
                self.plog(f'Task {prev_stage + 1} Concept Accuracy: {stage_metric.concept_accu}')
                self.plog(f'Task {prev_stage + 1} Class Accuracy: {stage_metric.clf_accu}')

                stage_concept_accuracies.append(stage_metric.concept_accu)
                stage_class_accuracies.append(stage_metric.clf_accu)
                csvwriter.writerow([prev_stage + 1, stage_metric.concept_accu, stage_metric.clf_accu])
                
                # 最后一个stage了---算了所有stage都测吧
                # if prev_stage == stage:
                if prev_stage == stage:
                    ## 全部都测一次，完整的concept和完整的label：就是让prev_class_ratio=0
                    test_loader = DataLoader(dataset=IncrementalIMGDataset(self.data_path, split='test', class_ratio=class_ratio, prev_class_ratio=0, concept_ratio=concept_ratio), batch_size=self.batch_size, shuffle=True, num_workers=16)
                    # 初始化一个新的 Metric 对象用于完整测试
                    full_test_metric = Metric(concept_mode='binary', clf_mode='multi')
                    # 遍历完整测试数据集
                    for data in tqdm.tqdm(test_loader, postfix=f'Full Test on Stage {stage + 1}'):
                        for i, item in enumerate(data):
                            data[i] = item.to(self.device)  # 确保所有输入张量都在 self.device 上
                        img, concept, label, one_hot_label = data
                        with torch.no_grad():
                            concept_pred, label_pred = self.model(img, num_concepts=self.model.num_concepts)
                            concept_subset = concept[:, :self.model.num_concepts]
                            full_test_metric.add(concept_pred, label_pred, concept_subset, label, one_hot_label)
                    
                    # 打印完整测试结果
                    print(f'Full Test on Stage {stage + 1} Concept Accuracy: {full_test_metric.concept_accu}')
                    print(f'Full Test on Stage {stage + 1} Class Accuracy: {full_test_metric.clf_accu}')
                    self.plog(f'Full Test on Stage {stage + 1} Concept Accuracy: {full_test_metric.concept_accu}')
                    self.plog(f'Full Test on Stage {stage + 1} Class Accuracy: {full_test_metric.clf_accu}')
                    
                    # # 将完整测试结果保存到最后一个stage的metrics.csv中
                    # csv_path = join(os.getcwd(), self.args.saved_dir, self.time, f'stage_{stage+1}_metrics.csv')
                    # with open(csv_path, mode='a', newline='') as csvfile:
                    #     csvwriter = csv.writer(csvfile)
                    #     csvwriter.writerow(['Full Test', full_test_metric.concept_accu, full_test_metric.clf_accu])

                    # 将完整测试结果保存到最后一个stage的metrics.csv中
                    csvwriter.writerow(['Full Test', full_test_metric.concept_accu, full_test_metric.clf_accu])



        avg_loss = loss / len(self.test_loader)
        self.test_losses.append(avg_loss)
        self.test_accus.append(metric.clf_accu)
        self.concept_accuracies.append(metric.concept_accu)
        self.class_accuracies.append(metric.clf_accu)
        self.stage_concept_accuracies[stage] = stage_concept_accuracies
        self.stage_class_accuracies[stage] = stage_class_accuracies
        self.plog(f'concept accu: {metric.concept_accu}')
        self.plog(f'classification accu: {metric.clf_accu}')
        print('concept accu: ', metric.concept_accu)
        print('classification accu: ', metric.clf_accu)

        # 计算遗忘率
        if stage > 0:
            for task in range(stage + 1):
                if task < stage:
                    if task < len(self.stage_concept_accuracies[stage]):
                        concept_forgetting_rate = max(self.stage_concept_accuracies[:stage][task]) - self.stage_concept_accuracies[stage][task]
                        class_forgetting_rate = max(self.stage_class_accuracies[:stage][task]) - self.stage_class_accuracies[stage][task]
                        concept_forgetting_rates.append(concept_forgetting_rate)
                        class_forgetting_rates.append(class_forgetting_rate)
                        self.plog(f'Task {task + 1} Concept Forgetting Rate: {concept_forgetting_rate}')
                        self.plog(f'Task {task + 1} Class Forgetting Rate: {class_forgetting_rate}')
                        print(f'Task {task + 1} Concept Forgetting Rate: ', concept_forgetting_rate)
                        print(f'Task {task + 1} Class Forgetting Rate: ', class_forgetting_rate)
                    else:
                        concept_forgetting_rates.append(0.0)
                        class_forgetting_rates.append(0.0)
                        self.plog(f'Task {task + 1} Concept Forgetting Rate: 0.0')
                        self.plog(f'Task {task + 1} Class Forgetting Rate: 0.0')
                        print(f'Task {task + 1} Concept Forgetting Rate: 0.0')
                        print(f'Task {task + 1} Class Forgetting Rate: 0.0')
                else:
                    concept_forgetting_rates.append(0.0)
                    class_forgetting_rates.append(0.0)
                    self.plog(f'Task {task + 1} Concept Forgetting Rate: 0.0')
                    self.plog(f'Task {task + 1} Class Forgetting Rate: 0.0')
                    print(f'Task {task + 1} Concept Forgetting Rate: 0.0')
                    print(f'Task {task + 1} Class Forgetting Rate: 0.0')

        self.concept_forgetting_rates.append(concept_forgetting_rates)
        self.class_forgetting_rates.append(class_forgetting_rates)

        return metric.clf_accu



    def plot_accuracy_curves(self):
        plt.figure(figsize=(10, 5))
        
        # 绘制每个stage的概念和类别准确率
        for stage in range(self.num_stages):
            plt.plot(self.stage_concept_accuracies[stage], label=f'Stage {stage+1} Concept Accuracy')
            plt.plot(self.stage_class_accuracies[stage], label=f'Stage {stage+1} Class Accuracy')
        
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.title('Concept and Class Accuracy Curves')
        plt.legend()
        plt.savefig(join(os.getcwd(), self.args.saved_dir, self.time, 'accuracy_curves.png'))
        plt.show()

    def plot_forgetting_curves(self):
        plt.figure(figsize=(10, 5))
        
        # 绘制每个stage的概念和类别遗忘率
        for stage in range(1, self.num_stages):
            plt.plot(self.concept_forgetting_rates[stage], label=f'Stage {stage+1} Concept Forgetting Rate')
            plt.plot(self.class_forgetting_rates[stage], label=f'Stage {stage+1} Class Forgetting Rate')
        
        plt.xlabel('Task')
        plt.ylabel('Forgetting Rate')
        plt.title('Concept and Class Forgetting Rate Curves')
        plt.legend()
        plt.savefig(join(os.getcwd(), self.args.saved_dir, self.time, 'forgetting_curves.png'))
        plt.show()


    def save_metrics_to_txt(self, overall_concept_accuracy, overall_class_accuracy, concept_forgetting_rate_mean, class_forgetting_rate_mean):
        txt_path = join(os.getcwd(), self.args.saved_dir, self.time, 'metrics.txt')
        with open(txt_path, mode='w') as file:
            file.write('Metrics\n')
            file.write(f'Overall Concept Accuracy: {overall_concept_accuracy}\n')
            file.write(f'Overall Class Accuracy: {overall_class_accuracy}\n')
            file.write(f'Concept Forgetting Rate Mean: {concept_forgetting_rate_mean}\n')
            file.write(f'Class Forgetting Rate Mean: {class_forgetting_rate_mean}\n')
            file.write('\n')
            file.write('Stage Concept Accuracy and Class Accuracy\n')
            for stage in range(self.num_stages):
                file.write(f'Stage {stage+1} Concept Accuracy: {self.concept_accuracies[stage]}\n')
                file.write(f'Stage {stage+1} Class Accuracy: {self.class_accuracies[stage]}\n')
            file.write('\n')
            file.write('Stage Concept Forgetting Rate and Class Forgetting Rate\n')
            for stage in range(1, self.num_stages):
                file.write(f'Stage {stage+1} Concept Forgetting Rate: {self.concept_forgetting_rates[stage-1]}\n')
                file.write(f'Stage {stage+1} Class Forgetting Rate: {self.class_forgetting_rates[stage-1]}\n')


    def save_stage_metrics_to_txt(self, stage, concept_accuracy, class_accuracy):
        txt_path = join(os.getcwd(), self.args.saved_dir, self.time, f'stage_{stage+1}_metrics.txt')
        with open(txt_path, mode='w') as file:
            file.write(f'Stage {stage+1} Metrics\n')
            file.write(f'Concept Accuracy: {concept_accuracy}\n')
            file.write(f'Class Accuracy: {class_accuracy}\n')

    def calculate_mean_metrics(self):
        overall_concept_accuracy = sum(self.concept_accuracies) / len(self.concept_accuracies)
        overall_class_accuracy = sum(self.class_accuracies) / len(self.class_accuracies)
        overall_concept_forgetting_rate = sum(sum(self.concept_forgetting_rates, [])) / len(sum(self.concept_forgetting_rates, []))
        overall_class_forgetting_rate = sum(sum(self.class_forgetting_rates, [])) / len(sum(self.class_forgetting_rates, []))
        self.plog(f'Overall Concept Accuracy: {overall_concept_accuracy}')
        self.plog(f'Overall Class Accuracy: {overall_class_accuracy}')
        self.plog(f'Overall Concept Forgetting Rate: {overall_concept_forgetting_rate}')
        self.plog(f'Overall Class Forgetting Rate: {overall_class_forgetting_rate}')
        print(f'Overall Concept Accuracy: {overall_concept_accuracy}')
        print(f'Overall Class Accuracy: {overall_class_accuracy}')
        print(f'Overall Concept Forgetting Rate: {overall_concept_forgetting_rate}')
        print(f'Overall Class Forgetting Rate: {overall_class_forgetting_rate}')

        # 保存总体结果到CSV文件
        csv_path = join(os.getcwd(), self.args.saved_dir, self.time, 'overall_metrics.csv')
        with open(csv_path, mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Metric', 'Value'])
            csvwriter.writerow(['Overall Concept Accuracy', overall_concept_accuracy])
            csvwriter.writerow(['Overall Class Accuracy', overall_class_accuracy])
            csvwriter.writerow(['Overall Concept Forgetting Rate', overall_concept_forgetting_rate])
            csvwriter.writerow(['Overall Class Forgetting Rate', overall_class_forgetting_rate])

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='----------Incremental CBM  baseline experiments----------')
    parser.add_argument('-seed', type=int, default=42, help='random seed')
    parser.add_argument('-dataset', type=str, default='cub', help='dataset_name')
    parser.add_argument('-batch_size', '-b', type=int, default=128, help='mini-batch size')
    parser.add_argument('-epoch', '-e', type=int, default=30, help='epochs for each training stage')
    parser.add_argument('-learning_rate', '-lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-gamma', type=float, default=0.95, help='decay for learning scheduler')
    parser.add_argument('-concept_lambda', type=float, default=0.5, help='concept lambda')
    parser.add_argument('-v_backbone', type=str, default='resnet', help='vision backbone, resnet or vit')
    parser.add_argument('-saved_dir', type=str, default='results', help='path to save results')
    parser.add_argument('-class_ratio', type=float, default=0.5, help='initial ratio of classes used')
    parser.add_argument('-concept_ratio', type=float, default=0.5, help='initial ratio of concepts used')
    parser.add_argument('-num_stages', type=int, default=6, help='number of incremental stages')
    args = parser.parse_args()
    check_dir(args.saved_dir)
    trainer = IncrementalIMGTrainer(parser=parser)
    trainer.train_incrementally()