"""Utility classes and methods
"""

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class SQuAD(Dataset):
    """Stanford Question and Answering Dataset (SQuAD).

    Each item in the dataset is a dictionary with the following entries:
        - context_idxs: Indices of the words in the context.
            Shape (context_len,)
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len)
        - question_idxs: Indices of words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of the start of the answer.
        - y2: Index of the end of the anser.
        - id: ID of the example.

    Args:
        datapath (str): Path to .npz file containing pre-processed dataset.
    """

    def __init(self, datapath):
        super(SQuAD, self).__init__()

        dataset = np.load(datapath)

        self.context_idxs = torch.tensor(dataset["context_idxs"], dtype=torch.long)
        self.context_char_idxs = torch.tensor(
            dataset["context_char_idxs"], dtype=torch.long
        )
        self.question_idxs = torch.tensor(dataset["ques_idxs"], dtype=torch.long)
        self.question_char_idxs = torch.tensor(
            dataset["ques_char_idxs"], dtype=torch.long
        )
        self.y1s = torch.tensor(dataset["y1s"], dtype=torch.long)
        self.y2s = torch.tensor(dataset["y2s"], dtype=torch.long)

        # Use index 0 for no-answer token (token 1 = OOV)
        ds_len, c_len, w_len = self.context_char_idxs.size()
        ones = torch.zeros((ds_len, 1), type=torch.long)
        self.context_idxs = torch.cat([ones, self.context_idxs], dim=1)
        self.question_idxs = torch.cat([ones, self.question_idxs], dim=1)

        ones = torch.zeros((ds_len, 1, w_len), dtype=torch.long)
        self.context_char_idxs = torch.cat([ones, self.context_char_idxs], dim=1)
        self.question_char_idxs = torch.cat([ones, self.question_char_idxs], dim=1)

        self.y1s += 1
        self.y2s += 1

        self.ids = torch.tensor(dataset["ids"], dtype=torch.long)

    def __getitem__(self, idx):
        sample = {
            "context_idxs": self.context_idxs[idx],
            "context_char_idxs": self.context_char_idxs[idx],
            "question_idxs": self.question_idxs[idx],
            "question_char_idxs": self.question_char_idxs[idx],
            "y1": self.y1s[idx],
            "y2": self.y2s[idx],
            "id": self.ids[idx],
        }

        return sample

    def __len__(self):
        return len(self.ids)


def masked_softmax(logits, mask=None, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over the given dimension and set
    entries to 0 where `mask` is 0.

    Passing `None` in for the mask is acceptable; you'll just get a
    regular softmax.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits` with zeros in positions
            that are also zero in the output.
        dim (int): Dimension over which to take the softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    if mask is None:
        probs = softmax_fn(logits, dim)
    else:
        masked_logits = logits.masked_fill(~mask, -float("inf"))
        probs = softmax_fn(masked_logits, dim)
    return probs
