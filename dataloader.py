# coding: utf-8
from pathlib import Path
import re

from transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

pattern = u'[^\u4e00-\u9fa50-9a-zA-Z]+'


class DataProcessor:
    def __init__(self, vocab_path: Path, max_length: int) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.max_length = max_length

    def readLCQMC(self, path: Path):
        data = open(path, 'r', encoding='utf-8')
        res = {}
        res['encodes'] = []
        res['labels'] = []
        for line in tqdm(data, desc='Loading data'):
            s1, s2, label = line.strip().split('\t')
            s1 = re.sub(pattern, '', s1)
            s2 = re.sub(pattern, '', s2)
            encodes = self.tokenizer.encode_plus(
                                                 s1,
                                                 text_pair=s2,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation="longest_first",
                                                 return_tensors='pt'
                                                 )
            res['encodes'].append(encodes)
            res['labels'].append(int(label))
        input_ids = torch.cat([item['input_ids'] for item in res['encodes']])
        attention_mask = torch.cat([item['attention_mask'] for item in res['encodes']])
        token_type_ids = torch.cat([item['token_type_ids'] for item in res['encodes']])
        labels = torch.LongTensor([item for item in res['labels']])
        # print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels.shape)
        return TensorDataset(input_ids, attention_mask, token_type_ids, labels)


class MyDataLoader:
    def __init__(self, vocab_path: Path, max_length: int) -> None:
        self.data_processor = DataProcessor(vocab_path, max_length)

    def load(self, path: Path, batch_size: int):
        data = self.data_processor.readLCQMC(path)
        sampler = RandomSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return data_loader
