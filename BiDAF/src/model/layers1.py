"""Assortment of layer to use in models.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer without the character-level component.

    Embedding vectors are further refined through a two-layer Highway Encoder
    (see `HighwayEncoder` class for more details).

    Args:
        wordvecs (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Dropout probability
    Returns:
        emb (torch.Tensor): Embedded representation of words.
    """

    def __init__(self, wordvecs, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        self.embed = nn.Embedding.from_pretrained(wordvecs)
        self.proj = nn.Linear(wordvecs.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(num_layers=2, hidden_size=hidden_size)

    def forward(self, x):
        # Shape: (batch_size, seq_len, embed_size)
        emb = self.embed(x)
        emb = F.dropout(emb, self.drop_prob, self.training)

        # Shape: (batch_size, seq_len, hidden_size)
        emb = self.proj(emb)

        # Shape: (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers
        hidden_size (int): Size of hidden activations
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.transforms = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, x: (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x), self.training)
            x = g * t + (1 - g) * x

        return x

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        # Initialize transform layer weights
        for transform in self.transforms:
            nn.init.kaiming_uniform_(transform.weight, nonlinearity="relu")
            nn.init.uniform_(transform.bias, -stdv, stdv)

        # Initialize gate bias to negative value to default to pass through
        for gate in self.gates:
            nn.init.kaiming_uniform_(gate.weight, nonlinearity="relu")
            nn.init.constant_(gate.bias, -2.0)


class RNNEncoder(nn.Module):
    """General purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each timestep.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    Returns:
        enc_output (torch.Tensor): Encoded output tensor of shape
            (batch_size, seq_len, hidden_size * 2)
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.0):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_prob if num_layers > 1 else 0.0,
        )

    def forward(self, x, lengths):
        # original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by lengths
        lengths, sort_idx = lengths.sort(dim=0, descending=True)
        x = x[sort_idx]

        # Pack sequence for RNN
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Shape: (batch_size, seq_len, 2 * hidden_size)
        enc_output, _ = self.rnn(x_packed)

        # Unpack and reverse sort
        enc_output, _ = pad_packed_sequence(
            enc_output, batch_first=True, total_len=orig_len
        )
        _, unsort_idx = sort_idx.sort(0)
        # Shape: (batch_size, seq_len, 2 * hidden_size)
        enc_output = enc_output[unsort_idx]

        # Apply dropout (RNN applies dropout after all but the last layer)
        enc_output = F.dropout(enc_output, self.drop_prob, self.training)

        return enc_output


class BiDAFAttention(nn.Module):
    """Bidirectional attention used by BiDAF.

    Bidirectional attention computes attention going in two directions:
    from context to question and from question to context. The output of
    this layer is the concatention of [context, c2q_attention, context *
    c2q_attention, context * q2c_attention]. This concatenation allows the
    attention vector at each timestep to flow through the attention layer
    to the modeling layer. The output has shape (batch_size, context_len,
    8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size

        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def forward(self, c, q, c_mask, q_mask):
        """Takes a mini-batch of context and question hidden states and outputs
        a combination of Context-to-Question (C2Q) attention and Question-to-Context
        (Q2C) attention.

        Args:
            c (torch.Tensor): Mini-batch of context hidden state tensors.
                Shape: (batch_size, c_len, 2 * hidden_size).
            q (torch.Tensor): Mini-batch of question hidden state tensors.
                Shape: (batch_size, q_len, 2 * hidden_size).
        Returns:
            g (torch.Tensor): 
        """
        c_len = c.size(1)
        q_len = q.size(1)

        # Shape: (batch_size, c_len, q_len)
        s = self.get_similarity(c, q)
        # Shape: (batch_size, c_len, 1)
        c_mask = c_mask.view(-1, c_len, 1)
        # Shape: (batch_size, 1, q_len)
        q_mask = q_mask.view(-1, 1, q_len)

        # Shape: (batch_size, c_len, q_len)
        s1 = masked_softmax(s, c_mask, dim=2)
        # Shape: (batch_size, c_len, q_len)
        s2 = masked_softmax(s, q_mask, dim=1)

        # C2Q attention. Shape: (bs, c_len, q_len) x (bs, , 2 * h) --> (bs, , 2 * h)
        a = torch.bmm(s1, q)

        # Q2C attention. Shape:
        b = torch.bmm(s2, c)

    def get_similarity_matrix(self, c, q):
        pass

    def reset_parameters(self):
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform(weight)
        nn.init.constant_(self.bias, 0.0)
