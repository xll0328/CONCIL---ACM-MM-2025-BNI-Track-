from torch import nn
from torch.nn import init
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, ViTModel

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


class NLPBaseModel(nn.Module):

    def __init__(self, num_concepts,  num_classes, num_concepts_classes):
        super().__init__()
        hidden_size = 128
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_concepts, self.num_concepts_classes = num_concepts, num_concepts_classes
        self.concept_fc = nn.Linear(self.bert.config.hidden_size, num_concepts * num_concepts_classes)
        self.final_fc = nn.Sequential(
                            nn.Linear(num_concepts * num_concepts_classes, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, num_classes)
        )
        self._initialize_weights()
    
    def forward(self, input_ids, attention_mask):
        b_size = input_ids.shape[0]
        hidden_output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        concept = self.concept_fc(hidden_output.last_hidden_state.mean(dim = 1))                     #class_token 
        label = self.final_fc(concept)

        return concept.reshape(b_size * self.num_concepts, self.num_concepts_classes), label
    
    def _initialize_weights(self):

        init.kaiming_uniform_(self.concept_fc.weight, nonlinearity='relu')
        init.constant_(self.concept_fc.bias, 0)
        for layer in self.final_fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0)
