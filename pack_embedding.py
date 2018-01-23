import numpy as np
from numpy import *
import torch.nn as nn
import torch

class LoadEmbedding(nn.Embedding):
    def __init__(self,num_embeddings,embedding_dim):
        super(LoadEmbedding, self).__init__(num_embeddings,embedding_dim)
        self.embedding_dict = {}

    def load_embedding_by_file(self,file,binary = False,encoding='utf8',datatype=float32):
        self.embedding_dict = {}
        with open(file,'rb') as fin:
            header = str(fin.readline(),encoding).split()
            if header.__len__()==2:
                vocab_size,dim_size = int(header[0]),int(header[1])
            else:
                vocab_size = fin.readlines().__len__()+1
                dim_size = header[1:].__len__()
                fin.seek(0)
            if binary:
                binary_len = dtype(datatype).itemsize*int(dim_size)
                for i in range(vocab_size):
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError
                        if ch != b'\n':
                            word.append(ch)
                    word = str(b''.join(word),encoding)
                    weight = fromstring(fin.read(binary_len), dtype=datatype)
                    self.embedding_dict[word] = weight
            else:
                for i in range(vocab_size):
                    data = str(fin.readline(),encoding).strip().split(' ')
                    word,weight = data[0],fromstring(' '.join(data[1:]),dtype=datatype)
                    self.embedding_dict[word] = weight
        narray = np.empty(0,0)
        for e in self.embedding_dict.values():
            np.concatenate(narray,e)
        self.weight = nn.Parameter(torch.FloatTensor(narray))
        self.num_embeddings,self.embedding_dim= vocab_size,dim_size
