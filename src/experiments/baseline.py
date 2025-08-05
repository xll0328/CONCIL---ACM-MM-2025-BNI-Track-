import os
import sys
sys.path.append(os.getcwd())

import torch
import tqdm
import argparse
from os.path import join
from torch.nn.utils import clip_grad_norm_
from src.utils.metrics import Metric
from src.utils.util import check_dir
from src.experiments.template import BASETrainer

class IMGTrainer(BASETrainer):

    def __init__(
            self,
            parser: argparse.ArgumentParser
    ):
        super().__init__(parser)

    def loss(self, img, concept, label, concept_lambda):

        concept_pred, label_pred = self.model(img)        
        concept_pred = torch.sigmoid(concept_pred)
        concept_loss = self.bce(concept_pred, concept)
        class_loss = self.ce(label_pred, label)

        return concept_loss * concept_lambda + class_loss

    def train_step(
            self,
            img,
            concept,
            label,
            concept_lambda,
    ):

        loss = self.loss(img, concept, label, concept_lambda)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):
        best_accu = 0
        for epoch in range(self.epoch):
            epoch_loss = []
            self.model.train()
            for data in tqdm.tqdm(self.train_loader, postfix = f'Training Epoch {epoch}'):
                for i, item in enumerate(data):
                    data[i] = item.to(self.device)

                img, concept, label, _ = data

                loss = self.train_step(
                    img,
                    concept,
                    label,
                    concept_lambda = self.args.concept_lambda,
                )
                
                # print('now loss is:',loss.item())
                epoch_loss.append(loss.item())
                
            # self.scheduler.step()
            print(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            self.plog(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            clf_accu = self.test()
            if clf_accu > best_accu:
                best_accu = clf_accu
                self.plog(f'Best checkpoint update, save ckpt at Epoch {epoch}\n')
                path_to_checkpoint = join(os.getcwd(), self.args.saved_dir, 
                                          self.time, 'best_ckpt.pth')
                torch.save(self.model, path_to_checkpoint)
            self.scheduler.step()
    
    def test(self):

        metric = Metric(concept_mode = 'binary', clf_mode = 'multi')
        loss = 0
        for data in tqdm.tqdm(self.test_loader, postfix = 'Testing Epoch'):
            for i, item in enumerate(data):
                data[i] = item.to(self.device)
            img, concept, label, one_hot_label = data
            with torch.no_grad():
                concept_pred, label_pred = self.model(img)
                concept_pred = torch.sigmoid(concept_pred)
                loss += (self.bce(concept_pred, concept) * self.args.concept_lambda + self.ce(label_pred, label)).item()
                metric.add(concept_pred, label_pred, concept, label, one_hot_label)

        self.plog(f'epoch testing loss: {loss / len(self.test_loader)}')
        self.plog(f'concept accu: {metric.concept_accu}')
        self.plog(f'classification accu: {metric.clf_accu}')
        self.plog(f'mean classification recall: {metric.clf_recall}')
        self.plog(f'mean classification precision: {metric.clf_precision}')
        self.plog(f'mean classification F1: {metric.clf_f1}\n')
        
        print('testing loss: ', loss / len(self.test_loader))
        print('concept accu: ', metric.concept_accu)
        print('classification accu: ', metric.clf_accu)
        print('mean classification recall: ', metric.clf_recall)
        print('mean classification precision: ', metric.clf_precision)
        print('mean classification F1: ', metric.clf_f1)
        
        return metric.clf_accu

class NLPTrainer(BASETrainer):
    
    def __init__(
            self,
            parser: argparse.ArgumentParser
    ):
        super().__init__(parser)

    def loss(self, input_ids, attention_mask, concept, label, concept_lambda):

        concept_pred, label_pred = self.model(input_ids, attention_mask)       
        assert concept_pred.shape[0] == concept.shape[0] * concept.shape[1]
        concept = concept.reshape((concept_pred.shape[0], ))
        concept_loss = self.ce(concept_pred, concept)
        class_loss = self.ce(label_pred, label)

        return concept_loss * concept_lambda + class_loss

    def train_step(
            self,
            input_ids,
            attention_mask,
            concept,
            label,
            concept_lambda,
    ):

        loss = self.loss(input_ids, attention_mask, concept, label, concept_lambda)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def train(self):
        
        best_accu = 0
        for epoch in range(self.epoch):
            epoch_loss = []
            self.model.train()
            with tqdm.tqdm(self.train_loader, desc=f'Training Epoch {epoch}') as pbar:
                for data in pbar:
                    for i, item in enumerate(data):
                        data[i] = item.to(self.device)

                        input_ids, attention_mask, concept, label, _ = data

                    loss = self.train_step(
                        input_ids,
                        attention_mask,
                        concept,
                        label,
                        concept_lambda = self.args.concept_lambda,
                )
                
                    epoch_loss.append(loss.item())
                    pbar.set_postfix({'loss': loss.item()})

            print(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            self.plog(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            clf_accu = self.test()
            if clf_accu > best_accu:
                best_accu = clf_accu
                self.plog(f'Best checkpoint update, save ckpt at Epoch {epoch}\n')
                path_to_checkpoint = join(os.getcwd(), self.args.saved_dir, 
                                          self.time, 'best_ckpt.pth')
                torch.save(self.model, path_to_checkpoint)

    def test(self):

        self.model.eval()
        if self.dataset == 'cebab':
            metric = Metric(concept_mode = 'multi', clf_mode = 'multi')
        elif self.dataset == 'imdb':
            metric = Metric(concept_mode = 'multi', clf_mode = 'binary')

        loss = 0

        with tqdm.tqdm(self.test_loader, desc = 'Testing Epoch') as pbar:
            for data in pbar:
                for i, item in enumerate(data):
                    data[i] = item.to(self.device)

                input_ids, attention_mask, concept, label, one_hot_label = data

                with torch.no_grad():
                    concept_pred, label_pred = self.model(input_ids, attention_mask)
                    concept = concept.reshape((concept_pred.shape[0], ))
                    concept_loss = self.ce(concept_pred, concept)
                    class_loss = self.ce(label_pred, label)
                    loss += (concept_loss * self.args.concept_lambda + class_loss).item()
                    
                    if metric.clf_mode == 'multi':
                        metric.add(concept_pred, label_pred, concept, label, one_hot_label)
                    else:
                        metric.add(concept_pred, label_pred, concept, label)

        self.plog(f'epoch testing loss: {loss / len(self.test_loader)}')
        self.plog(f'concept accu: {metric.concept_accu}')
        self.plog(f'classification accu: {metric.clf_accu}')
        self.plog(f'mean classification recall: {metric.clf_recall}')
        self.plog(f'mean classification precision: {metric.clf_precision}')
        self.plog(f'mean classification F1: {metric.clf_f1}\n')

        print('testing loss: ', loss / len(self.test_loader))
        print('concept accu: ', metric.concept_accu)
        print('classification accu: ', metric.clf_accu)
        print('mean classification recall: ', metric.clf_recall)
        print('mean classification precision: ', metric.clf_precision)
        print('mean classification F1: ', metric.clf_f1)
        
        return metric.clf_accu


if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = '----------baseline experiments----------')

    parser.add_argument('-seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-dataset', type = str, default = 'cub', help = 'dataset_name')
    parser.add_argument('-batch_size', '-b', type = int, default= 64, help='mini-batch size')
    parser.add_argument('-epoch', '-e', type = int, default = 50, help='epochs for training process')
    parser.add_argument('-learning_rate', '-lr', type = float, default = 1e-4, help="learning rate")
    parser.add_argument('-weight_decay', type = float, default = 5e-5, help='weight decay for optimizer')
    parser.add_argument('-gamma', type = float, default = 0.95, help='decay for learning scheduler')
    parser.add_argument('-concept_lambda', type = float, default = 0.5, help = 'concept lambda')
    parser.add_argument('-v_backbone', type = str, default = 'resnet', help = 'vision backbone, resnet or vit')
    parser.add_argument('-saved_dir', type = str, default = 'results', help = 'path to save results')

    args = parser.parse_args()
    description = parser.description
    check_dir(args.saved_dir)
    if args.dataset == 'cub' or args.dataset == 'awa':
        trainer = IMGTrainer(parser = parser)
    else:
        trainer = NLPTrainer(parser = parser)
    trainer.train()
