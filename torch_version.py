import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1) 
lstm = nn.LSTM(3, 3)
inputs = [torch.randn(1, 3) for _ in range(5)] #(5, 1, 3)

# #hidden state
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
# for i in inputs:
#     out, hidden = lstm(i.view(1, 1, -1), hidden)

# print(inputs)
inputs = torch.cat(inputs).view(len(inputs), 1, -1) #concat 5 tensors to a large tensor
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)

#prepare_data
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
tag_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
# print(word_to_ix)
# print(tag_to_ix)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

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
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

