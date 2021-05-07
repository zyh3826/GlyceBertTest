# coding: utf-8
import os
import time
from typing import Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
# from glyce.models.glyce_bert.glyce_bert_classifier import GlyceBertClassifier
from glyce.utils.optimization import BertAdam
from glyce.dataset_readers.bert_config import Config
# from transformers import BertConfig
try:
    from apex import amp
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex to use fp16 training.'
    )

from toolFunction.trainer.trainer import Trainer
from toolFunction.trainer.yaml_config import CfgNode
from dataloader import MyDataLoader
from model import GlyceBertClassifier


class MyTrainer(Trainer):
    def __init__(self, args: CfgNode, logger) -> None:
        super().__init__(args, logger)
        self.data_loader = MyDataLoader(args.model.model_path, args.datasets.max_length)

    def get_train_dataloader(self):
        self.logger.info('Loading train data')
        train_iter = self.data_loader.load(self.args.datasets.train_path, self.args.training.batch_size)
        return train_iter

    def get_eval_dataloader(self):
        self.logger.info('Loading eval data')
        eval_iter = self.data_loader.load(self.args.datasets.eval_path, self.args.training.batch_size)
        return eval_iter

    def get_test_dataloader(self):
        self.logger.info('Loading test data')
        test_iter = self.data_loader.load(self.args.datasets.test_path, self.args.training.batch_size)
        return test_iter

    def init_model(self):
        path = os.path.join(self.args.model.model_path, 'bert_config.json')
        config = Config.from_json_file(path)
        # config = BertConfig.from_pretrained(self.args.model.model_path)
        self.model = GlyceBertClassifier(config, num_labels=self.args.training.num_labels)
        if self.args.model.continue_training:
            self.logger.info('Loading continue training state_dict...')
            self.model.load_state_dict(torch.load(self.args.model.continue_training_path))
        self.model.to(self.args.model.device)
        self.model.train()

    def create_optimizer(self, num_train_steps):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.args.optimizer.lr,
                                  warmup=self.args.scheduler.warmup_prob,
                                  t_total=num_train_steps)

    def get_inputs(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        input_ids, attention_mask, token_type_ids, labels = batch
        inputs = {
            'input_ids': input_ids.cuda(self.args.model.device),
            'attention_mask': attention_mask.cuda(self.args.model.device),
            'token_type_ids': token_type_ids.cuda(self.args.model.device)
            }
        return inputs, labels

    def train(self):
        ''' Train the model '''
        self.seed_everything()
        self.init_model()
        self.create_loss_fn()
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        test_dataloader = self.get_test_dataloader()

        if self.args.training.max_steps > 0:
            t_total = self.args.training.max_steps
            self.args.training.num_train_epochs = self.args.training.max_steps // (len(train_dataloader) // self.args.training.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.training.gradient_accumulation_steps * self.args.training.num_train_epochs

        # create optimizer
        self.create_optimizer(num_train_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        tail = time.strftime('%m-%d_%H.%M', time.localtime()) + '_{}_seed_{}'.format(self.args.training.model_type, self.args.training.seed) + '_best_model'
        self.args.training.best_save_path = os.path.join(self.args.training.save_path, tail)
        if not os.path.exists(self.args.training.best_save_path):
            os.makedirs(self.args.training.best_save_path)

        # fp16
        if self.args.model.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.args.model.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            self.logger.info('***** Initialize Data Parallel *****')
            gpus = [int(item) for item in self.args.model.gpus.split()]
            model = nn.DataParallel(self.model, device_ids=gpus, output_device=gpus[0])
            self.args.model.device = gpus[0]
            model.to(self.args.model.device)

        # tenorboardx
        log_dir = self.init_log_writer()

        # Train!
        self.logger.info('***** Running training *****')
        self.logger.info('  Num examples = %d', len(train_dataloader))
        self.logger.info('  Num Epochs = %d', self.args.training.num_train_epochs)
        self.logger.info('  Instantaneous batch size = %d', self.args.training.batch_size)
        self.logger.info('  Gradient Accumulation steps = %d', self.args.training.gradient_accumulation_steps)
        self.logger.info('  Total optimization steps = %d', t_total)

        global_step = 0
        self.dev_best_loss = float('inf')
        self.model.zero_grad()

        for epoch in range(int(self.args.training.num_train_epochs)):
            desc = 'Training. Epoch: {}/{}'.format(epoch+1, int(self.args.training.num_train_epochs))
            for step, batch in enumerate(tqdm(train_dataloader, desc=desc)):
                loss_item, outputs, labels = self.training_step(batch, global_step, epoch)
                if self.args.training.do_adv:
                    self.do_adv(batch)
                if (step + 1) % self.args.training.gradient_accumulation_steps == 0:
                    if self.args.model.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.args.training.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.args.training.max_grad_norm)
                    self.optimizer.step()
                    self.model.zero_grad()
                global_step += 1
                if self.args.training.logging_steps > 0 and global_step % self.args.training.logging_steps == 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    self.log(eval_dataloader, loss_item, outputs, labels, global_step, lr)
            self.logger.info('\n')
            if 'cpu' not in str(self.args.model.device):
                torch.cuda.empty_cache()
        self.predict(test_dataloader)
        self.logger.info('***** Finish training *****')
        self.logger.info('  TensorBoardX log at {}'.format(log_dir))
        self.logger.info('  Best model save at {}'.format(self.args.training.best_save_path))

    def training_step(self, batch, global_step: int, idx: int):
        inputs, labels = self.get_inputs(batch)
        outputs, glyph_loss = self.model(**inputs)
        loss = self.compute_loss(outputs, labels.cuda(self.args.model.device))
        # model outputs are always tuple in pytorch-transformers (see doc)
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            # mean() to average on multi-gpu parallel training
            loss = loss.mean()
            glyph_loss = glyph_loss.mean()

        if self.args.training.gradient_accumulation_steps > 1:
            loss = loss / self.args.training.gradient_accumulation_steps
            glyph_loss = glyph_loss / self.args.training.gradient_accumulation_steps

        if global_step < self.args.training.glyph_warmup:
            sum_loss = loss + self.args.training.glyph_ratio * glyph_loss
        else:
            sum_loss = loss + self.args.training.glyph_ratio * glyph_loss * self.args.training.glyph_decay ** (idx + 1)

        if self.args.model.fp16:
            with amp.scale_loss(sum_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            sum_loss.backward()

        loss_item = sum_loss.item()
        return loss_item, outputs, labels

    def evaluate_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, float]:
        inputs, labels = self.get_inputs(batch)
        outputs, _ = self.model(**inputs)
        loss = self.compute_loss(outputs, labels.cuda(self.args.model.device))
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            #  mean() to average on multi-gpu parallel evaluating
            loss = loss.mean()
        return outputs, labels, loss.item()


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO, format=u'%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    config = CfgNode()
    config.merge_from_file('./config.yaml')
    trainer = MyTrainer(config, logger)
    trainer.train()
