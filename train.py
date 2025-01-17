from transformer_model import SingleLayerTransformer
import pandas as pd
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
model=SingleLayerTransformer(5625, 50, 64, 5625, 4)
corpus=pd.read_pickle('corpus')
vocab=pd.read_pickle('vocab')
wtoi={w: i for i,w in enumerate(vocab)}
itow={i: w for i,w in enumerate(vocab)}
corpus_list=corpus.split()
def encode(text):
    if isinstance(text, str):
        _text=text.split()
    elif isinstance(text, list):
        _text=text
    encoding=torch.tensor([wtoi[w.lower()] for w in _text])
    return encoding

def decode(indices):
    text=[itow[i] for i in indices]
    s=''
    for x in text:
        s+=x
        s+=" "
    return s

def get_batch(batch_size=8,context_length=64):
    idx_list=[random.randint(0,len(corpus_list)-context_length-1) for i in range(batch_size)]
    batch=torch.stack([encode(corpus_list[idx:idx+context_length]) for idx in idx_list])
    y=torch.stack([encode(corpus_list[idx+1:idx+1+context_length]) for idx in idx_list])
    return (batch,y)
learning_rate=1e-3
n_steps=15000
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_list=[]
ppl=[]
for i in range(n_steps):
    x,y=get_batch()
    logits=model(x)
    B,T,D=logits.shape
    loss=F.cross_entropy(logits.reshape(B*T,D), y.reshape(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(np.exp(loss.detach().numpy()))
    if i%100==0:
        ppl.append(np.mean(loss_list[-100:]))
        print(np.mean(loss_list[-100:]))
plt.plot(ppl)
plt.title("PPL per Token")
plt.xlabel("Iterations/100")
plt.savefig("PPL.png")

for i in range(5):
    x,y=get_batch()
    indices=model.generate(x)
    model.get_attn_visual(x)
    index_list=indices.squeeze().tolist()
    # print(index_list)
    print(decode(index_list[0]))
    print("____________________")