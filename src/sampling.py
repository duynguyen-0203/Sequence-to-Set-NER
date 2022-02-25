import torch
from src import utils


def create_train_sample(doc):
    """
    Create a training sample for the model

    :param doc: document to build training sample
    :return:
        encoding: token indices, numerical representations of tokens building the sequences that will be used as input
        by the BERT model
        pos_encoding: input of POS embedding
        wordvec_encoding: the list of vocabulary id of tokens
        context_masks: attention mask for the BERT model
        token_masks_bool: mask the number of tokens in the document
        token_masks: mask the encoding position of each token in the document's encoding
        char_encoding: input of LSTM model for character embedding
        token_masks_char: mask the encoding of each character in the document's character level encoding
        char_count: the list of number of characters in each token (include <EOT> character)
        gold_entity_spans_token: inner span token of each gold entity
        gold_entity_types: list of entity types of each gold entity
        gold_entity_masks: mask each span is gold entity or not (useful when the document doesn't have a gold entity)
    """
    encoding = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encoding)

    list_char_encoding = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in list_char_encoding:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token, dtype=torch.long))
    char_encoding = utils.padded_stack(char_encoding)
    token_masks_char = (char_encoding != 0).long()
    char_count = torch.tensor(char_count, dtype=torch.long)

    pos_encoding = [token.pos for token in doc.tokens]
    wordvec_encoding = [token.vocab_id for token in doc.tokens]

    token_masks = []
    for token in doc.tokens:
        token_masks.append(create_mask(*token.span, context_size))

    gold_entity_spans_token = []
    gold_entity_types = []
    gold_entity_masks = []
    for entity in doc.entities:
        gold_entity_spans_token.append(entity.span_token)
        gold_entity_types.append(entity.entity_type.id)
        gold_entity_masks.append(1)

    encoding = torch.tensor(encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    if len(gold_entity_types) > 0:
        gold_entity_spans_token = torch.tensor(gold_entity_spans_token, dtype=torch.long)
        gold_entity_spans_token[:, 1] = gold_entity_spans_token[:, 1] - 1
        gold_entity_types = torch.tensor(gold_entity_types, dtype=torch.long)
        gold_entity_masks = torch.tensor(gold_entity_masks, dtype=torch.bool)
    else:
        gold_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gold_entity_types = torch.zeros([1], dtype=torch.long)
        gold_entity_masks = torch.tensor([1], dtype=torch.bool)

    return dict(encoding=encoding, pos_encoding=pos_encoding, wordvec_encoding=wordvec_encoding,
                context_masks=context_masks, token_masks_bool=token_masks_bool, token_masks=token_masks,
                char_encoding=char_encoding, token_masks_char=token_masks_char, char_count=char_count,
                gold_entity_spans_token=gold_entity_spans_token, gold_entity_types=gold_entity_types,
                gold_entity_masks=gold_entity_masks)


def create_eval_sample(doc):
    """
    Create the sample for evaluation phase

    :param doc: document to build the evaluation sample
    :returns: dictionary
        - dictionary - asf
        - bfdl - fd
        encoding: token indices, numerical representations of tokens building the sequences that will be used as input
        by the BERT model
        pos_encoding: input of POS embedding
        wordvec_encoding: the list of vocabulary id of tokens
        context_masks: attention mask for the BERT model
        token_masks_bool: mask the number of tokens in the document
        token_masks: mask the encoding position of each token in the document's encoding
        char_encoding: input of LSTM model for character embedding
        token_masks_char: mask the encoding of each character in the document's character level encoding
        char_count: the list of number of characters in each token (include <EOT> character)
    """
    encoding = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encoding)

    list_char_encoding = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in list_char_encoding:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token, dtype=torch.long))
    char_encoding = utils.padded_stack(char_encoding)
    token_masks_char = (char_encoding != 0).long()
    char_count = torch.tensor(char_count, dtype=torch.long)

    pos_encoding = [token.pos for token in doc.tokens]
    wordvec_encoding = [token.vocab_id for token in doc.tokens]

    token_masks = []
    for token in doc.tokens:
        token_masks.append(create_mask(*token.span, context_size))

    encoding = torch.tensor(encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    return dict(encoding=encoding, pos_encoding=pos_encoding, wordvec_encoding=wordvec_encoding,
                context_masks=context_masks, token_masks_bool=token_masks_bool, token_masks=token_masks,
                char_encoding=char_encoding, token_masks_char=token_masks_char, char_count=char_count)


def create_mask(left, right, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[left:right] = 1

    return mask
