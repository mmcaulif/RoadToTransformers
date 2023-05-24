import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.datasets import AG_NEWS

from embeddings import collate_batch, VOCAB_LEN

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
        print(embedded.shape)
        import sys
        sys.exit()
        
        # Num heads is 1 rn
        q = self.w_q(embedded).unsqueeze(1) # [N, 1, L, d_k]
        k = self.w_k(embedded).unsqueeze(1) # [N, 1, L, d_k]
        v = self.w_v(embedded) # [N, 1, L, d_k]
        
        q_k = th.bmm(q, k.transpose(1, 2)).squeeze(-1) # [N, 1, L, L]

        q_k = F.softmax(q_k/math.sqrt(self.d_k), dim=-1)    # [N, 1, L, L]
        v_q_k = q_k * v # [N, 1, L, d_k]
        # print(v_q_k.shape)    # [N, 1, L*d_k] # Need to flatten out to be passed through linear layer
        out = self.linear(v_q_k)
        return F.softmax(out, dim=-1)


"""def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with th.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets).argmax(1).item()
            total_acc += int(predicted_label == label.argmax(1).item())
            total_count += 1

    return total_acc/total_count"""

train_iter, test_iter = AG_NEWS()
dataloader = DataLoader(train_iter, batch_size=1024, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=1, shuffle=True, collate_fn=collate_batch)

att_model = SelfAttentionLayer(64, 64, VOCAB_LEN)
optimizer = th.optim.SGD(att_model.parameters(), lr=10)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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

    att_model.eval()
    total_acc, total_count = 0, 0

    with th.no_grad():
        for idx, (label, text, offsets) in enumerate(test_dataloader):
            predicted_label = att_model(text, offsets).argmax(1).item()
            total_acc += int(predicted_label == label.argmax(1).item())
            total_count += 1

    eval_acc = total_acc/total_count

    # scheduler.step()
    print(f"Epoch {e+1} complete, average loss: {sum_loss/(idx+1)}, eval accuracy: {eval_acc}\n")


"""
Need to implement and understand embeddings with sequences etc.

Links:
    embeddings
    1. https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    2. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    3. https://www.baeldung.com/cs/transformer-text-embeddings
    4. https://www.baeldung.com/cs/ml-word2vec-topic-modeling
    5. https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    text classification
    1. https://towardsdatascience.com/text-classification-with-pytorch-7111dae111a6
    2. https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

    attention/transformers
    1. https://ar5iv.labs.arxiv.org/html/1706.03762
    2. https://machinelearningmastery.com/the-transformer-attention-mechanism/
    3. https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    4. https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html <- detailed
    5. https://spotintelligence.com/2023/01/31/self-attention/
    6. https://github.com/datnnt1997/multi-head_self-attention/blob/master/selfattention.py


"""