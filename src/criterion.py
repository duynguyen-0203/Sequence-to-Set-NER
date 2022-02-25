import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment


class Criterion(nn.Module):
    def __init__(self, entity_types, matcher, weight_dict, losses):
        super(Criterion, self).__init__()
        self.entity_types = entity_types
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_class(self, outputs, targets, indices, num_spans):
        """Classification loss (cross entropy loss)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        targets['labels'] = targets['labels'].split(targets['sizes'], dim=-1)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['labels'], indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        empty_weight = torch.ones(src_logits.size(-1), device=src_logits.device)
        empty_weight[0] = num_spans / (src_logits.size(0) * src_logits.size(1) - num_spans)
        loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)

        losses = {'loss_class': loss_class}

        return losses

    def loss_boundary(self, outputs, targets, indices, num_spans):
        """Boundary loss (negative log likelihood loss)"""
        assert 'pred_boundaries' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_spans_left = outputs['pred_boundaries'][0][idx]
        src_spans_right = outputs['pred_boundaries'][1][idx]

        targets['spans'] = targets['spans'].split(targets['sizes'], dim=0)
        target_spans = torch.cat([t[i] for t, (_, i) in zip(targets['spans'], indices)], dim=0)

        src_spans_left_logp = torch.log(1e-30 + src_spans_left)
        src_spans_right_logp = torch.log(1e-30 + src_spans_right)

        left_nll_loss = F.nll_loss(src_spans_left_logp, target_spans[:, 0], reduction='none')
        right_nll_loss = F.nll_loss(src_spans_right_logp, target_spans[:, 1], reduction='none')

        loss_boundary = left_nll_loss + right_nll_loss

        losses = {}
        losses['loss_boundary'] = loss_boundary.sum() / num_spans

        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_premuatation_idx(indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'class': self.loss_class,
            'boundary': self.loss_boundary
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_spans = sum(targets['sizes'])
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans, **kwargs))

        return losses


class HugarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_span: float = 1):
        super(HugarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        assert cost_class != 0 or cost_span != 0

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            out_prob = outputs['pred_logits'].flatten(0, 1).softmax(dim=-1)

            entity_left = outputs['pred_boundaries'][0].flatten(0, 1)
            entity_right = outputs['pred_boundaries'][1].flatten(0, 1)

            tgt_ids = targets['labels']
            tgt_spans = targets['spans']

            cost_class = -out_prob[:, tgt_ids]
            cost_span = -(entity_left[:, tgt_spans[:, 0]] + entity_right[:, tgt_spans[:, 1]])

            C = self.cost_span * cost_span + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            sizes = targets['sizes']
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # import pdb; pdb.set_trace()
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
