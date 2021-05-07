# coding: utf-8
import torch.nn as nn
from torch.nn import CrossEntropyLoss


from glyce.layers.classifier import SingleLinearClassifier, MultiNonLinearClassifier
# from glyce.layers.glyce_transformer import GlyceTransformer
from glyce_transformer import GlyceTransformer


class GlyceBertClassifier(nn.Module):
    def __init__(self, config, num_labels=2):
        super(GlyceBertClassifier, self).__init__()
        self.num_labels = num_labels
        self.glyph_transformer = GlyceTransformer(config, num_labels=num_labels)
        # config involves here
        # 1. config.glyph_config
        # 2. config.bert_config
        # 3. transformer_config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # print('model on: {}'.format(next(self.glyph_transformer.parameters()).device))
        encoded_layers, _, glyph_cls_loss = self.glyph_transformer(
                                                                input_ids,
                                                                token_type_ids=token_type_ids,
                                                                attention_mask=attention_mask)

        features_output = encoded_layers[-1]
        # print(features_output.shape)
        features_output = features_output[:, 0, :]

        features_output = self.dropout(features_output)
        logits = self.classifier(features_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, glyph_cls_loss
        else:
            return logits, glyph_cls_loss
