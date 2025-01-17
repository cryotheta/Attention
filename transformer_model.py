import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.head_dim = head_dim

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        tril=torch.tril(torch.ones(x.shape[1] ,x.shape[1] ))
        
        attn =  Q@K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        attn = attn.masked_fill(tril == 0, float('-inf'))
        attention_weights = F.softmax(attn, dim=-1)
        
        # Compute the output
        output = torch.matmul(attention_weights, V)
        return output
    def inspect_attention(self,x):
        with torch.no_grad():
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
            
            # Compute attention scores
            tril=torch.tril(torch.ones(x.shape[1],x.shape[1] ))
            
            attn =  Q@K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
            attn = attn.masked_fill(tril == 0, float('-inf'))
            attention_weights = F.softmax(attn, dim=-1)
            return attention_weights

class SingleLayerTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, head_dim, output_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.initialize_glove_embeddings()
        self.attention_heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.W_o = nn.Linear(num_heads*head_dim,embed_dim)
        self.unembedding = nn.Linear(embed_dim, output_dim)
        pe = torch.stack([torch.tensor([np.cos(j/128**(2*i/embed_dim)) if j%2==1 else np.sin(j/128**(2*i/embed_dim)) for j in range(64)]) for i in range(embed_dim)])
        self.register_buffer('pe', pe)
    def initialize_glove_embeddings(self):
        self.embedding.weight.data.copy_(torch.load('glove_embeddings.pt')).to(dtype=torch.float32)
    def forward(self, x):
        # print(self.pe.shape)
        embeddings = self.embedding(x)+self.pe.transpose(-2,-1).to(dtype=torch.float32)
        attention_outputs = [head(embeddings) for head in self.attention_heads]
        concatenated_outputs = torch.cat(attention_outputs, dim=-1)  
        transformed_output=self.W_o(concatenated_outputs)+embeddings  #residual connection
        logit = self.unembedding(transformed_output) 
        return logit
    def generate(self,x,max_tokens=16):
        for i in range(max_tokens):
            x=x[:,-64:]
            logit=self.forward(x)
            next_token_logits=logit[:,-1,:]
            next_token=torch.multinomial(F.softmax(next_token_logits,dim=-1),1)
            x=torch.cat((x,next_token), dim=-1)
        return x
    # def beam_search(slef,x,beam_size=4,max_tokens=16):
    #     for data in range(x.shape[0]):
    #         scores=[]
    #         indices=[]
    #         for i in range(max_tokens):
    #             x=x[data,-64:]
    #             logit=self.forward(x)
    #             next_token_logits=logit[:,-1,:]
    #             next_score, next_index=torch.topk(F.softmax(next_token_logits,dim=-1),beam_size)
    #             if len(scores)==0:
    #                 scores.append(next_token)
    #                 indices.append(next_index)
    #             else:
    #                 big_scores=[]
    #                 for i in scores:
    #                     for j in next_score:
    #                         big_scores.append(i+j)
    #                 best_score, best_index=torch.topk(torch.tensor(big_scores))
    #                 for i in best_index:
    #                     indices[i//beam_size].append(i%beam_size)
    #                     score[i//beam_size]=big_scores[i]
    #             x=torch.cat((x,next_token), dim=-1)

    def get_attn_visual(self,x):
        vocab=pd.read_pickle('vocab')
        wtoi={w: i for i,w in enumerate(vocab)}
        itow={i: w for i,w in enumerate(vocab)}
        with torch.no_grad():
            embeddings = self.embedding(x)+self.pe.transpose(-2,-1).to(dtype=torch.float32)
            for idx, head in enumerate(self.attention_heads):
                attn_=head.inspect_attention(embeddings)
                print(attn_.shape)
                for batch in range(x.shape[0]):
                    G = nx.Graph()
                    plt.figure(figsize=(16, 16))
                    labels=[itow[int(w.item())] for w in x[batch,:]]
                    G.add_nodes_from(labels)                   

                    heatmap=attn_[batch,:,:].numpy()
                    flat_indices = np.argsort(heatmap, axis=None)[-15:]
                    indices_2d = np.unravel_index(flat_indices, heatmap.shape)
                    # print(indices_2d)
                    G.add_edges_from([(labels[i],labels[j]) for i,j in zip(indices_2d[0],indices_2d[1])])
                    # pos = {}
                    # pos.update((node, (1, 2*index)) for index, node in enumerate(labels))  # Place set A on x=1
                    # pos.update((node, (100,2* index+1)) for index, node in enumerate(labels))  # Place set B on x=2
                    plt.figure(figsize=(20, 20))
                    nx.draw(G,  pos=nx.shell_layout(G), with_labels=True, node_size=400, node_color='skyblue', font_size=24, edge_color='gray')

                    # Set labels and title
                    plt.title("Sparse Bipartite Graph between Two Sets", fontsize=14)
                    plt.savefig(f"attn_plots_large/Map_{idx}_{batch}")
                    plt.clf()

                    sns.heatmap(heatmap,xticklabels=labels, yticklabels=labels)
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    plt.savefig(f"attn_plots_large/heatmap{idx}_sentence_{batch}.png")
                    plt.clf()


        


    
