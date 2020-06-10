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
        emb = F.dropout(emb, self.drop_prob)

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
        self.transforms = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # Initialize gate bias to negative value to default to pass through
        for gate in self.gates:
            nn.init.constant_(gate.bias, -2.0)

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, x: (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    def __init__(self, )