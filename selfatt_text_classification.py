import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
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

text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: F.one_hot(th.as_tensor(int(x) - 1).long(), num_classes=4)
label_pipeline = lambda x: np.eye(4)[int(x) - 1]

"""print(label_pipeline("2"))
import sys
sys.exit()"""

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_k, d_v, vocab_size, emb_size=64, n_classes=4) -> None:
        super(SelfAttentionLayer, self).__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.embedding = nn.EmbeddingBag(vocab_size, emb_size, sparse=False)

        self.w_q = nn.Linear(emb_size, self.d_k)
        self.w_k = nn.Linear(emb_size, self.d_k)
        self.w_v = nn.Linear(emb_size, self.d_v)
        self.linear = nn.Linear(self.d_v, n_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        q = self.w_q(embedded).unsqueeze(1)
        k = self.w_k(embedded).unsqueeze(1)
        v = self.w_v(embedded)

        # print(q.shape, k.permute(0, 2, 1).shape)
        
        q_k = th.matmul(q, k.permute(0, 2, 1)).squeeze(-1)

        # print(q_k.shape, v.shape)

        q_k = q_k/math.sqrt(self.d_k)
        v_q_k = F.softmax(q_k, dim=1) * v
        out = self.linear(v_q_k)
        # print(out)
        out = F.softmax(out, dim=1)
        # print(out)
        return out
    

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

def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with th.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets).argmax(1).item()
            total_acc += int(predicted_label == label.argmax(1).item())
            total_count += 1

    return total_acc/total_count

train_iter, test_iter = AG_NEWS()
dataloader = DataLoader(train_iter, batch_size=512, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=1, shuffle=True, collate_fn=collate_batch)

att_model = SelfAttentionLayer(64, 64, len(vocab))
optimizer = th.optim.SGD(att_model.parameters(), lr=5)

for e in range(10):
    sum_loss = 0
    for idx, (label, text, offsets) in tqdm(enumerate(dataloader)):
        # print(idx, label, text)

        optimizer.zero_grad()
        out = att_model(text, offsets)
        # print(out.shape, label.shape)
        loss = F.cross_entropy(out, label)

        with th.no_grad():
            sum_loss += loss

        loss.backward()
        th.nn.utils.clip_grad_norm_(att_model.parameters(), 0.5)
        optimizer.step()

    print(f"Epoch {e+1} complete, average loss: {sum_loss/(idx+1)}, eval accuracy: {evaluate(att_model, test_dataloader)}\n")


"""
Links:
    embeddings
    1. https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    2. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

    text classification
    1. https://towardsdatascience.com/text-classification-with-pytorch-7111dae111a6
    2. https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

    attention/transformers
    1. https://ar5iv.labs.arxiv.org/html/1706.03762
    2. https://machinelearningmastery.com/the-transformer-attention-mechanism/
    3. https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    4. https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html <- detailed
    5. https://spotintelligence.com/2023/01/31/self-attention/


"""