# Copyright (c) 2023 Chuanze Lu
import logging

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math


class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.act = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        return self.w_2(self.dropout(self.act(self.w_1(xs))))


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_head: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        assert input_dim % num_head == 0

        self.d_k = input_dim // num_head
        self.h = num_head
        self.linear_q = torch.nn.Linear(input_dim, input_dim)
        self.linear_k = torch.nn.Linear(input_dim, input_dim)
        self.linear_v = torch.nn.Linear(input_dim, input_dim)
        self.linear_out = torch.nn.Linear(input_dim, input_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor = torch.empty(0)
                )-> torch.Tensor:

        batch_size = value.size(0)
        q, k, v = self.forward_qkv(query, key, value)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            score = score.masked_fill(mask, -float('inf'))
            attention = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            attention = torch.softmax(score, dim=-1)

        attention = self.dropout(attention)
        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linear_out(x)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb

        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 self_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True):

        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(input_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(input_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x = residual + self.dropout(self.self_attn(x, x, x, mask))

        residual = x

        if not self.normalize_before:
            x = self.norm1(x)

            residual = x

        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
                 :param lengths:
                 :param max_len:
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class TransformerEncoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 pos_enc_layer_type: str = 'abs_pos',
                 normalize_before: bool = True,
                 padding_idx: int = -1
                 ):

        super(TransformerEncoder, self).__init__()

        if pos_enc_layer_type == 'abs_pos':
            pos_enc = PositionalEncoding
        else:
            pos_enc = None
        # pos_enc = PositionalEncoding

        self.embedding = nn.Sequential(
            nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            pos_enc(output_size, positional_dropout_rate)
        )

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadAttention(output_size, attention_heads, attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                dropout_rate,
                normalize_before
            ) for i in range(num_blocks)
        ])

        self.after_norm = nn.LayerNorm(output_size, eps=1e-12)
        self.normalize_before = normalize_before

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        T = x.size(1)
        mask = ~make_pad_mask(x_lens, T).unsqueeze(1)
        # x = self.embedding[0](x)
        x, pos_emb = self.embedding(x)

        for block in self.encoders:
            x, mask = block(x, mask)

        if self.normalize_before:
            x = self.after_norm(x)

        return x, mask


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, embed_size, attention_heads, num_layers, dropout):
        super(TransformerLM, self).__init__()

        self.encoder = TransformerEncoder(vocab_size,
                                          embed_size,
                                          attention_heads,
                                          linear_units=2048,
                                          num_blocks=num_layers,
                                          dropout_rate=dropout)

 #       self.rnn = nn.LSTM(input_size=embed_size,
 #                         hidden_size=300,
 #                          num_layers=1,
 #                          bidirectional=False,
 #                          dropout=dropout,
 #                          batch_first=True)

        self.project = nn.Linear(embed_size, vocab_size)

    def forward(self, input: torch.Tensor, input_lens: torch.Tensor):

        x, mask = self.encoder(input, input_lens)

        #xo, _ = self.rnn(x)

        logits = self.project(x)

        return logits
