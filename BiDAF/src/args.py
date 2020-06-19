"""Command-line arguments for make_dataset.py, train.py, test.py"""

import argparse


def get_setup_args():
    """Get command-line arguments for `make_dataset.py`."""
    parser = argparse.ArgumentParser("Download and pre-process SQuAD")

    add_common_args(parser)

    # SQuAD v2.0
    parser.add_argument(
        "--train_url",
        type=str,
        default="https://github.com/chrischute/squad/data/train-v2.0.json",
    )
    parser.add_argument(
        "--dev_url",
        type=str,
        default="https://github.com/chrischute/squad/data/dev-v2.0.json",
    )
    parser.add_argument(
        "--test_url",
        type=str,
        default="https://github.com/chrischute/squad/data/test-v2.0.json",
    )

    # GloVe
    parser.add_argument(
        "--glove_url",
        type=str,
        default="http://nlp.stanford.edu/data/glove.840B.300d.zip",
    )
    parser.add_argument(
        "--glove_dim", type=int, default=300, help="Size of GloVe word vectors to use"
    )
    parser.add_argument(
        "--glove_num_vecs", type=int, default=2196017, help="Number of GloVe vectors"
    )

    # Tokenization
    parser.add_argument(
        "--word2idx_file", type=str, default="../data/processed/word2idx.json"
    )
    parser.add_argument(
        "--char2idx_file", type=str, default="../data/processed/char2idx.json"
    )

    args = parser.parse_args()

    return args


def add_common_args(parser):
    """Add arguments common to all three scripts: `make_dataset.py`, `train.py`, `test.py`."""

    # Record files
    parser.add_argument(
        "--train_record_file", type=str, default="../data/processed/train.npz"
    )
    parser.add_argument(
        "--dev_record_file", type="str", default="../data/processed/dev.npz"
    )
    parser.add_argument(
        "--test_record_file", type="str", default="../data/processed/test.npz"
    )

    # Word and character embeddings
    parser.add_argument(
        "--word_emb_file", type="str", default="../data/processed/word_emb.json"
    )
    parser.add_argument(
        "--char_emb_file", type="str", default="../data/processed/char_emb.json"
    )

    # TODO: Evaluation files
    parser.add_argument(
        "--train_eval_file", type=str, default="../data/processed/train_eval.json"
    )
    parser.add_argument(
        "--dev_eval_file", type=str, default="../data/processed/dev_eval.json"
    )
    parser.add_argument(
        "--test_eval_file", type=str, default="../data/processed/test_eval.json"
    )
