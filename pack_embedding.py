import numpy as np
from numpy import *
import torch.nn as nn
import torch
import pickle
import os
import random

random.seed(5)


class LoadEmbedding(nn.Embedding):
    def __init__(self, num_embeddings,embedding_dim,hyperparameter):
        super(LoadEmbedding, self).__init__(num_embeddings,embedding_dim)
        self.V = num_embeddings
        self.D = embedding_dim
        self.embedding_dict = {}
        if hyperparameter.pretrain:
            self.load_pretrained_embedding(hyperparameter.pretrain_file,hyperparameter.vocab,hyperparameter.embed_pickle,
                                           hyperparameter.pretrain_file_bianry)
        else:
            self.init_embedding()

    def load_pretrained_embedding(self, file, vocab_dict, embed_pickle=None, binary=False,
                                  requires_grad = False,encoding='utf8', datatype=float32):
        """
        :param file: pretrained embedding file path
        :param vocab_dict: features dict
        :param embed_pickle: save embed file
        :param binary: if the file is binary ,set binary True,else set False
        :param requires_grad: fine-tuned
        :param encoding: the default encoding is 'utf8'
        :param datatype: vector datatype , the default is float32
        """
        if embed_pickle is None:
            raise FileNotFoundError
        if os.path.exists(embed_pickle):
            nparray = pickle.load(open(embed_pickle, 'rb'))
            vec_sum = np.sum(nparray[0:nparray.shape[0] - 1, :], axis=0)
            nparray[nparray.shape[0] - 1] = vec_sum / (nparray.shape[
                                                           0] - 1)  # -unknown- vector initialize by making average, -unknown- index is the last one
            print("vocabulary complete...")
            self.weight = nn.Parameter(torch.FloatTensor(nparray),requires_grad=requires_grad)
        else:
            with open(file, 'rb') as fin:
                header = str(fin.readline(), encoding).split()
                vocab_size = dim_size = 0
                if binary:
                    if header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        print("don't support this type")
                        exit(0)
                    binary_len = dtype(datatype).itemsize * int(dim_size)
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
                        word = str(b''.join(word), encoding)
                        weight = fromstring(fin.read(binary_len), dtype=datatype)
                        if word in vocab_dict:
                            self.embedding_dict[word] = weight
                else:
                    if header.__len__() == 1:
                        dim_size = int(header[0])
                        vocab_size = fin.readlines().__len__() + 1
                        fin.seek(0)
                    elif header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        vocab_size = fin.readline().__len__() + 1
                        dim_size = header[1:].__len__()
                        fin.seek(0)
                    for i in range(vocab_size):
                        data = str(fin.readline(), encoding).strip().split(' ')
                        word, weight = data[0], fromstring(' '.join(data[1:]), dtype=datatype, sep=' ')
                        if word in vocab_dict:
                            self.embedding_dict[word] = weight

            nparray = np.zeros(shape=[self.V,self.D])
            num = 0
            for k, v in vocab_dict.items():
                if k in self.embedding_dict.keys():
                    nparray[v] = np.array(self.embedding_dict[k])
                elif v == 0:
                    nparray[v] = np.array([[0 for i in range(dim_size)]])
                else:
                    nparray[v] = np.array([[random.uniform(-0.01, 0.01) for i in range(dim_size)]])
                num += 1
                # print("word : {}".format(k))
            print("vocabulary complete...")
            vec_sum = np.sum(nparray[0:nparray.shape[0] - 1, :], axis=0)
            nparray[nparray.shape[0] - 1] = vec_sum / (nparray.shape[
                                                           0] - 1)  # -unknown- vector initialize by making average, -unknown- index is the last one
            pickle.dump(nparray, open(embed_pickle, 'wb'))
            self.weight = nn.Parameter(torch.FloatTensor(nparray),requires_grad=requires_grad)

    def init_embedding(self,embed_pickle = None,requires_grad = False,init_way=0):
        """

        :param embed_pickle: pretrained embedding file path
        :param requires_grad: fine-tuned
        :param init_way: 0: init by zero. 1: init by normal. 2: init by uniform
        :return:
        """
        if embed_pickle is None:
            raise FileNotFoundError
        if os.path.exists(embed_pickle):
            nparray = pickle.load(open(embed_pickle, 'rb'))
            print("vocabulary complete...")
            self.weight = nn.Parameter(torch.FloatTensor(nparray), requires_grad=requires_grad)
        else:
            nparray = np.zeros(shape=[self.V,self.D])
            for v in range(self.V):
                if init_way == 0:
                    nparray[v] = np.array([[0 for i in range(self.D)]])
                elif init_way == 1:
                    nparray[v] = np.array([[random.normal(0,0.01) for i in range(self.D)]])
                elif init_way == 2:
                    nparray[v] = np.array([[random.uniform(-0.01,0.01) for i in range(self.D)]])
                else:
                    raise RuntimeError("init failed,pls check init way")
            pickle.dump(nparray,open(embed_pickle,'wb'))
            self.weight = nn.Parameter(torch.FloatTensor(nparray),requires_grad=requires_grad)

