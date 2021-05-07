# coding: utf-8
import torch
import torch.nn as nn
from glyce.layers.glyph_position_embed import GlyphPositionEmbedder
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler


class GlyceTransformer(nn.Module):
    def __init__(self, config, num_labels=4):
        super(GlyceTransformer, self).__init__()
        self.num_labels = num_labels
        self.glyph_embedder = GlyphPositionEmbedder(config.glyph_config)
        self.bert_model = BertModel.from_pretrained(config.glyph_config.bert_model)
        transformer_config = BertConfig.from_dict(config.transformer_config.to_dict())
        self.transformer_layer = BertEncoder(transformer_config)
        self.pooler = BertPooler(config)
        if config.bert_frozen == "true":
            print("!=!"*20)
            print("Please notice that the bert model if frozen")
            print("the loaded weights of models is ")
            print(config.glyph_config.bert_model)
            print("!-!"*20)
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        glyph_embed, glyph_cls_loss = self.glyph_embedder(input_ids, token_type_ids=token_type_ids)
        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'return_dict': True
        }
        outputs = self.bert_model(**inputs)
        sequence_output, pooled_output = outputs.hidden_states, outputs.pooler_output
        context_bert_output = sequence_output[-1]
        input_features = torch.cat([glyph_embed, context_bert_output], -1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        outputs = self.transformer_layer(input_features, extended_attention_mask, output_hidden_states=True)
        encoded_layers = outputs.hidden_states
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        return encoded_layers, pooled_output, glyph_cls_loss
