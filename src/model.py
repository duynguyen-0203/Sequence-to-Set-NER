import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import RobertaPreTrainedModel, RobertaConfig, RobertaModel


list_char = ['!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
             '8', '9', ':', ';', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '°',
             '²', '³', '¼', 'À', 'Á', 'Â', 'Ê', 'Í', 'Ð', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê',
             'ì', 'í', 'ð', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă', 'ă', 'Đ', 'đ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư',
             'Ạ', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'ẫ', 'Ậ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'Ặ', 'ặ', 'ẹ', 'ẻ',
             'ẽ', 'ế', 'ề', 'ể', 'Ễ', 'ễ', 'Ệ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ', 'ỗ', 'ộ',
             'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'ỡ', 'ợ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ',
             'ỵ', 'ỷ', 'ỹ', '–', '‘', '’', '“', '”', '…']


class DecoderLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, n_heads=8):
        r"""
        Initialization of a layer of Entity Set Decoder

        :param d_model: total dimension of the model
        :param d_ffn:
        :param dropout: dropout probability
        :param n_heads: number of parallel attention heads
        """
        super().__init__()

        # self attention
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout,
                                                    batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout,
                                                     batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # feedforward network
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.relu = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(self, tgt, src, mask):
        # self attention
        q = k = v = tgt
        tgt2 = self.self_attention(q, k, v)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        q = k = v = src
        tgt2 = self.cross_attention(q, k, v, key_padding_mask=~mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # feedforward network
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layers):
        super().__init()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, embed, src, mask):
        tgt = embed.unsqueeze(0).expand(src.size(0), -1, -1)
        output = tgt

        for layer in self.layers:
            output = layer(output, src, mask)

        return output


class Fuse(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, bert, query, mask):
        bert_embed = bert.unsqueeze(1).expand(-1, query.size(1), -1, -1)
        query_embed = query.unsqueeze(2).expand(-1, -1, bert.size(-2), -1)
        fuse = torch.cat([bert_embed, query_embed], dim=-1)
        x = self.W(fuse)

        x = self.v(torch.tanh(x)).squeeze(-1)
        mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x[~mask] = -1e25
        x = x.softmax(dim=-1)

        return x


class Sequence2Set(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, embed: torch.tensor, entity_types: int, prop_drop: float,
                 freeze_transformer: bool, num_decoder_layers: int = 3, lstm_layers: int = 3, lstm_drop: float = 0.4,
                 pos_size: int = 25,
                 char_lstm_layers: int = 1, char_lstm_drop: float = 0.2, char_size: int = 25, use_fasttext: bool = True,
                 use_pos: bool = True, use_char_lstm: bool = True, num_query: int = 60, reduce_dim: bool = False,
                 bert_before_lstm: bool = False):
        super(Sequence2Set, self).__init__(config)
        self.bert = RobertaModel(config)
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_fasttext = use_fasttext
        self.use_pos = use_pos
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_drop = char_lstm_drop
        self.char_size = char_size
        self.use_char_lstm = use_char_lstm
        self.reduce_dim = reduce_dim
        self.bert_before_lstm = bert_before_lstm

        lstm_input_size = 0
        if self.bert_before_lstm:
            lstm_input_size = config.hidden_size
        if use_fasttext:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size=char_size, hidden_size=char_size, num_layers=char_lstm_layers,
                                     bidirectional=True, dropout=char_lstm_drop, batch_first=True)
            self.char_embedding = nn.Embedding(len(list_char) + 3, char_size)

        if self.use_fasttext or self.use_pos or self.use_char_lstm or self.bert_before_lstm:
            lstm_hidden_size = lstm_input_size
            if self.bert_before_lstm:
                lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                                bidirectional=True, dropout=lstm_drop, batch_first=True)
            if self.reduce_dim and not self.bert_before_lstm:
                self.reduce_dimension = nn.Linear(2 * lstm_input_size + config.hidden_size, config.hidden_size)

        # Decode
        self.query_embed = nn.Embedding(num_query, config.hidden_size * 2)
        decoder_layer = DecoderLayer(d_model=config.hidden_size, d_ffn=1024, dropout=0.1)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)

        self.entity_classifier = nn.Linear(config.hidden_size, entity_types)
        self.entity_left = Fuse(config.hidden_size)
        self.entity_right = Fuse(config.hidden_size)

        self.dropout = nn.Dropout(prop_drop)
        self.entity_types = entity_types

        # weight initialization
        self.init_weights()
        if use_fasttext:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)

        if freeze_transformer:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks: torch.tensor,
                token_masks_bool: torch.tensor, pos_encoding: torch.tensor = None,
                wordvec_encoding: torch.tensor = None,
                char_encoding: torch.tensor = None, token_masks_char=None, char_count: torch.tensor = None,
                mode='train'):
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]

        batch_size = encodings.shape[0]
        token_count = token_masks_bool.long().sum(-1, keepdim=True)
        h_token = combine(h, token_masks, 'mean')

        embeds = []
        if self.bert_before_lstm:
            embeds = [h_token]

        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_fasttext:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)

        if self.use_char_lstm:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count * batch_size, max_char_count)
            char_encoding[char_count == 0][:, 0] = len(list_char) + 1  # <EOT> id
            char_count[char_count == 0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input=char_embed, lengths=char_count.tolist(),
                                                                  enforce_sorted=False, batch_first=True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = combine(char_embed, token_masks_char, 'mean')
            embeds.append(h_token_char)

        if len(embeds) > 0:
            h_token_pos_wordvec_char = torch.cat(embeds, dim=-1)
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(input=h_token_pos_wordvec_char,
                                                                                lengths=token_count.squeeze(
                                                                                    -1).cpu().tolist(),
                                                                                enforce_sorted=False, batch_first=True)
            h_token_pos_wordvec_char_packed_o, (_, _) = self.lstm(h_token_pos_wordvec_char_packed)
            h_token_pos_wordvec_char, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o,
                                                                           batch_first=True)

            rep = [h_token_pos_wordvec_char]
            if not self.bert_before_lstm:
                rep.append(h_token)
            h_token = torch.cat(rep, dim=-1)
            if self.reduce_dim and not self.bert_before_lstm:
                h_token = self.reduce_dimension(h_token)

        query_embed = self.query_embed.weight
        tgt = self.decoder(query_embed, h_token, token_masks_bool)

        entity_clf = self.entity_classifier(tgt)
        entity_left = self.entity_left(h_token, tgt, token_masks_bool)
        entity_right = self.entity_right(h_token, tgt, token_masks_bool)
        # import pdb; pdb.set_trace()

        return entity_clf, (entity_left, entity_right)


def combine(sub, sup_mask, pool_type='max'):
    sup = None
    if len(sub.shape) == len(sup_mask.shape):
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else:
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0

    return sup