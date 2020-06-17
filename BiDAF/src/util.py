"""Utility classes and methods
"""

import torch
import torch.nn.functional as F


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
