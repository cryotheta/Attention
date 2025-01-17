import pandas as pd 
import re
import torch
import pickle
df=pd.read_csv("Reviews_q2_main(1).csv")
print(df.head())
text=df['Text']
corpus=""
for i in text[:1500]:
    corpus+=str(i)
print(len(corpus))
vocab=[]
corpus=re.sub(r'[^A-Za-z0-9 ]+', '', corpus)
tokens=corpus.split(" ")
print(tokens[:5])
for i in tokens:
    if i.lower() in vocab:
        continue
    else:
        vocab.append(i.lower())
print(len(vocab))
print(vocab[:20])
glove_embeddings = {}
with open('glove.twitter.27B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab:
            vector = torch.tensor(list(map(float, values[1:])))
            glove_embeddings[word] = vector

# Create weight matrix
embedding_matrix = torch.zeros(len(vocab), 50)
num_embeddings=len(vocab)
for i, word in enumerate(glove_embeddings.keys()):
    if i >= num_embeddings:
        break
    embedding_matrix[i] = glove_embeddings[word]
print(embedding_matrix[:5])
print(embedding_matrix.shape)
torch.save(embedding_matrix,'glove_embeddings_large.pt')
with open("vocab_large", "wb") as fp:   #Pickling
    pickle.dump(vocab, fp)
with open("corpus_large", "wb") as fp:
    pickle.dump(corpus,fp)