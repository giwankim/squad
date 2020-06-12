import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    """Embedding layer without the character-level component.

    Embedding vectors are further refined through a two-layer Highway Encoder
    (see `HighwayEncoder` class for more details).

    Args:
        wordvecs (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Dropout probability
    """

    def __init__(self, wordvecs, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(wordvecs)
        self.proj = nn.Linear(wordvecs.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        # Shape: (batch_size, seq_len, embed_size)
        emb = self.embed(x)
        emb = F.dropout(emb, self.drop_prob, self.training)

        # Shape: (batch_size, seq_len, hidden_size)
        emb = self.proj(emb)
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

    Encoded output is the RNN's hidden state at each position that has shape
    `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
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
        enc_output = enc_output[unsort_idx]

        # Apply dropout
        enc_output = F.dropout(enc_output, self.drop_prob, self.training)

        return enc_output


class BiDAFAttention(nn.Module):
    """Bidirectional attention used by BiDAF.

    Bidirectional attention computes attention going in two directions:
    from context to question and from question to context. The output of
    this layer is the concatention of [context, ]

    Args:
        hidden_size (int):
        drop_prob (float):
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, c, q, c_mask, q_mask):
        pass

    def get_similarity_matrix(self, c, q):
        pass
