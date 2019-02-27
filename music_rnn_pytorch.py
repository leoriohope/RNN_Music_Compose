import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

'''data preparation
'''
def viewData(fileName):
    '''
    Input:
    fileName: str
    Output:
    length: int  #length of the music samle
    total: int   #totoal number of unit
    '''
    with open(fileName, 'r') as f:
        length = 0
        total = 0
        for line in f:
            length += 1
            total += len(line.split(" "))
    return length, total

length, total = viewData('music.txt')
# print(length)
# print(total // length)
n_X = length
len_X = total // length

def loadData(fileName):
    '''
    Input:
        filename : str
    Output:
        int_data: nparray (703)
    '''
    int_data = []
    with open(fileName, 'r') as f:
        for line in f:
            int_data.append([int(num) for num in line.split(' ')])
    return int_data

training_data = loadData('music.txt') #(703,)
print(len(training_data))

EMBEDDING_DIM = 10
HIDDEN_DIM = 64

#Define model
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #lstm take embedding as input and out put hidden state
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)  #The output y' using hidden as input , output tagset size vector
        self.hidden = self.init_hidden()
    
    def init_hidden(self):  #has to be 3D in pytorch setup
        return(torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))  #downgrade dimention   ? but why we don't set output dim
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#Training the model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 114, 114)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def prepare_sequence(seq):
    return torch.tensor(seq, dtype=torch.long)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0])
    tag_scores = model(inputs)
    print(tag_scores)

losses = []
for epoch in range(50):
    loss_one_epoch = 0
    print("epoch: " + str(epoch))
    for sentence in training_data[:100]:
        model.zero_grad()
        model.hidden = model.init_hidden()
        targets = sentence[-1:] + sentence[:-1]
        sentence_in = prepare_sequence(sentence)
        targets = prepare_sequence(targets)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss_one_epoch += loss.detach().numpy()
        loss.backward()
        optimizer.step()
    losses.append(loss_one_epoch)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0])
    tag_scores = model(inputs)
    print(tag_scores)

print(losses)
plt.plot(losses)
plt.show()


