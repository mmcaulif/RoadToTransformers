import numpy as np
import torch as th
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = iter(AG_NEWS(split='train'))

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text, )

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
VOCAB_LEN = len(vocab)
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: np.eye(4)[int(x) - 1]

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = th.tensor(text_pipeline(_text), dtype=th.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = th.from_numpy(np.array(label_list)).float()
    offsets = th.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = th.cat(text_list)

    return label_list, text_list, offsets