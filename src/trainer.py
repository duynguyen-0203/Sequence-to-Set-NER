import argparse
from datetime import datetime
import os
import sys
import logging

import torch

from src import utils


class Trainer:
    def __init__(self, args: argparse):
        self.args = args
        name = str(datetime.datetime.now()).replace(' ', '_')
        self._log_path = os.path.join(self.args.log_path, self.args.label, name)
        os.makedirs(self._log_path, exist_ok=True)

        self._save_path = os.path.join(self.args.save_path, self.args.label, name)
        os.makedirs(self._save_path, exist_ok=True)

        self._log_paths = dict()

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self._logger = logging.getLogger()
        utils.reset_logger(self._logger)

        file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)

        self._logger.setLevel(logging.INFO)

        self._summary_writer = SummaryWriter()
        self._best_results = dict()
        self._log_arguments()

        set_seed(args.seed)

        self._tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=False)

    def train(self, train_path, dev_path):
        args = self.args
        train_label, dev_label = 'train', 'dev'

        self._logger.info(f'Train dataset: {train_path}, Dev dataset: {dev_path}')

        # self._init_train_logging(train_label)
        # self._init_eval_logging(dev_label)

        reader = Reader(self._tokenizer, list_ner, list_pos, list_char, '/content/drive/MyDrive/NER/VLSP_2016')
        train_dataset = reader.read(train_path, 'VLSP_2016')
        dev_dataset = reader.read(dev_path, 'VLSP_2016')

        train_sample_count = len(train_dataset)
        updates_epoch = train_sample_count // args.batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info(f'Updates per epoch: {updates_epoch}')
        self._logger.info(f'Updates total: {updates_total}')

        config = RobertaConfig.from_pretrained(self.args.bert_path)
        embed = torch.from_numpy(reader.embedding_weight).float()
        # config: RobertaConfig, embed: torch.tensor, entity_types: int, prop_drop: float, freeze_transformer: bool, num_decoder_layers: int=3, lstm_layers: int=3, lstm_drop: float=0.4, pos_size: int=25,
        # char_lstm_layers: int=1, char_lstm_drop: float=0.2, char_size: int=25, use_fasttext: bool=True, use_pos: bool=True, use_char_lstm: bool=True, num_query: int=60, reduce_dim: bool=False, bert_before_lstm: bool=False
        model = Sequence2Set.from_pretrained(self.args.bert_path, config=config, embed=embed,
                                             entity_types=len(list_ner) + 1, prop_drop=self.args.prop_drop,
                                             freeze_transformer=self.args.freeze_transformer,
                                             num_decoder_layers=self.args.decoder_layers,
                                             lstm_layers=self.args.lstm_layers, lstm_drop=self.args.lstm_drop,
                                             pos_size=self.args.pos_size, char_lstm_layers=self.args.char_lstm_layers,
                                             char_lstm_drop=self.args.char_lstm_drop, char_size=self.args.char_size,
                                             use_fasttext=self.args.use_fasttext, use_pos=self.args.use_pos,
                                             use_char_lstm=self.args.use_char_lstm, num_query=self.args.num_query,
                                             bert_before_lstm=self.args.bert_before_lstm)
        model.cuda()

        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay, correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # entity_types, device, model, optimizer, scheduler, max_grad_norm
        compute_loss = ModelLoss(len(list_ner) + 1, 'cuda', model, optimizer, scheduler, args.max_grad_norm)

        best_f1 = 0.0
        for epoch in range(args.epochs):
            self._train_epoch(model, compute_loss, optimizer, train_dataset, epoch)
            self._eval(model, dev_dataset, epoch, confidence=self.args.confidence)

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer, dataset, epoch):
        self._logger.info(f'--------------------------Train epoch {epoch}--------------------------')

        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                 collate_fn=self._collate_fn)
        model.zero_grad()
        epoch_loss = 0.0
        for batch in tqdm(data_loader, total=len(data_loader), desc=f'Train epoch {epoch}'):
            model.train()
            batch = to_device(batch, 'cuda')
            import pdb;
            pdb.set_trace()

            entity_logits, entity_boundary = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                                   token_masks_bool=batch['token_masks_bool'],
                                                   token_masks=batch['token_masks'], pos_encoding=batch['pos_encoding'],
                                                   wordvec_encoding=batch['wordvec_encoding'],
                                                   char_encoding=batch['char_encoding'],
                                                   token_masks_char=batch['token_masks_char'],
                                                   char_count=batch['char_count'], mode='train')
            batch_loss = compute_loss.compute(entity_logits=entity_logits, entity_boundary=entity_boundary,
                                              entity_types=batch['gold_entity_types'],
                                              entity_spans_token=batch['gold_entity_spans_token'],
                                              entity_masks=batch['gold_entity_masks'])
            epoch_loss += batch_loss
        epoch_loss /= len(data_loader)
        self._logger.info(f'Loss: {epoch_loss}')
        print(epoch_loss)

        return epoch_loss

    def _eval(self, model: torch.nn.Module, dataset: Dataset, epoch, confidence: float = 0.5):
        # dataset: Dataset, tokenizer: AutoTokenizer, logger, no_overlapping: bool, epoch: int
        evaluator = Evaluator(dataset, self._tokenizer, self._logger, self.args.no_overlapping, epoch)
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                 collate_fn=self._collate_fn)

        with torch.no_grad():
            model.eval()

            for batch in tqdm(data_loader, total=len(data_loader), desc=f'Evaluate epoch {epoch}'):
                batch = to_device(batch, 'cuda')
                entity_logits, entity_boundary = model(encodings=batch['encodings'],
                                                       context_masks=batch['context_masks'],
                                                       token_masks_bool=batch['token_masks_bool'],
                                                       token_masks=batch['token_masks'],
                                                       pos_encoding=batch['pos_encoding'],
                                                       wordvec_encoding=batch['wordvec_encoding'],
                                                       char_encoding=batch['char_encoding'],
                                                       token_masks_char=batch['token_masks_char'],
                                                       char_count=batch['char_count'], mode='eval')

                evaluator.eval_batch(entity_logits, entity_boundary, confidence)

        ner_eval = evaluator.compute_scores()
        print(ner_eval)
        return ner_eval

    def _log_arguments(self):
        save_dict(self._log_path, self.args, 'args')

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, ]

        return optimizer_params

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()

        for key in keys:
            samples = [s[key] for s in batch]

            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                if key == 'encodings':
                    padded_batch[key] = padded_stack(samples, padding=self._tokenizer.pad_token_id)
                else:
                    padded_batch[key] = padded_stack(samples)

        return padded_batch