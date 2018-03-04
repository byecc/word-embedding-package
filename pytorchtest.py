import torch
from torch.autograd import Variable
from numpy import *

N,D=1,2
a = Variable(torch.FloatTensor(N,D),requires_grad = True)


