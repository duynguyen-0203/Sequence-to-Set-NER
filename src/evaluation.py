from typing import Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs

from transformers import AutoTokenizer
import torch

from src.entities import *


class Evaluator:
    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer, logger, no_overlapping: bool, epoch: int):
        self._tokenizer = tokenizer
        self._logger = logger
        self._dataset = dataset
        self._no_overlapping = no_overlapping
        self._epoch = epoch
        self._gt_entities = []
        self._pred_entities = []
        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_entity_boundary: torch.tensor, confidence: float):
        batch_size = batch_entity_clf.shape[0]

        batch_entity_clf = batch_entity_clf.softmax(dim=-1)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        batch_entity_scores = batch_entity_clf.max(dim=-1)[0]
        batch_entity_mask = batch_entity_scores > confidence

        entity_left = batch_entity_boundary[0].argmax(dim=-1)
        entity_right = batch_entity_boundary[1].argmax(dim=-1) + 1
        batch_entity_spans = torch.stack([entity_left, entity_right], dim=-1)

        batch_entity_mask = batch_entity_mask * (batch_entity_spans[:, :, 0] < batch_entity_spans[:, :, 1]) * (
                    batch_entity_types != 0)

        for i in range(batch_size):
            entity_mask = batch_entity_mask[i]

            entity_types = batch_entity_types[i][entity_mask]
            entity_scores = batch_entity_scores[i][entity_mask]
            entity_spans = batch_entity_spans[i][entity_mask]

            sample_pred_entities = self._convert_pred_entities(entity_types, entity_spans, entity_scores)
            sample_pred_entities = sorted(sample_pred_entities, key=lambda x: x[3], reverse=True)
            sample_pred_entities = self._remove_duplicate(sample_pred_entities)

            self._pred_entities.append(sample_pred_entities)

    def compute_scores(self):
        gt, pred = self._convert_by_settting(self._gt_entities, self._pred_entities, include_entity_types=True)
        # import pdb; pdb.set_trace()
        ner_eval = self._score(gt, pred, print_results=True)

        return ner_eval

    def _convert_by_settting(self, gt: List[List[Tuple]], pred: List[List[Tuple]], include_entity_types: bool = True,
                             include_score: bool = False):
        assert len(gt) == len(pred)

        def convert(t):
            c = list(t[:3])

            if include_score and len(t) > 3:
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.id)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.id)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results):
        print(gt_all)
        print(pred_all)
        labels = [t.id for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]

        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._dataset.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            cls_score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, cls_score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _remove_duplicate(self, entities):
        non_duplicate_entities = []

        for i, can_entity in enumerate(entities):
            find = False
            for j, entity in enumerate(non_duplicate_entities):
                if can_entity[0] == entity[0] and can_entity[1] == entity[1]:
                    find = True
            if not find:
                non_duplicate_entities.append(can_entity)
        return non_duplicate_entities

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        self._log(row_fmt % columns)

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            self._log(row_fmt % self._get_row(m, t.name))

        self._log('')

        # micro
        self._log(row_fmt % self._get_row(micro, 'micro'))

        # macro
        self._log(row_fmt % self._get_row(macro, 'macro'))

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_entities = doc.entities
            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple_token() for entity in gt_entities]

            if self._no_overlapping:
                sample_gt_entities = self._remove_overlapping(sample_gt_entities)

            self._gt_entities.append(sample_gt_entities)

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []
        for entity in entities:
            if not self._is_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)