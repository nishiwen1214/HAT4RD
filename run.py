import torch
import torch.nn as nn
import numpy as np

import torchtext
from torchtext.legacy.datasets import TranslationDataset
# from torchtext.data import Field, BucketIterator
from torchtext.legacy.data import Field
from torchtext.legacy.data import LabelField

import spacy
import random
import math
import os
import numpy as np

import time
import torch
from torchtext.data import *
from train_test import *
from model import Word_BiLSTM,Post_BiLSTM
import torch.optim as optim
from torchtext.legacy.data import Field,TabularDataset,Iterator,BucketIterator
CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


SEED = 2008
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True     #tell the GPU to use the CuDNN algorithem
torch.backends.cudnn.benchmark = False
TEXT = Field(tokenize='spacy')                  #we will put our text data in TEXT,and it will be tokened automaticly
                                                     #"spacy "is a NLP library ,can be used to tokenization
LABEL = LabelField(dtype=torch.float)           # Label 
fields = [ (None, None), ('text', TEXT), ('rumour', LABEL)]
# Load the data
print('Loading the data...')
train_data, valid_data, test_data = TabularDataset.splits(
                            path = 'data',
                            train = '/pheme2018_train.csv',
                            validation='/pheme2018_dev.csv',
                            test = '/pheme2018_test.csv',
                            format = 'csv',
                            fields = fields,
                            skip_header = True
)
print('building the vocabulary...')
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.300d")     #form the train_data building the vocabulary  #vectors: pre_tain the word vector:300__dimentions                                                                          
LABEL.build_vocab(train_data) 
# ——————————————————
# Hyper-parameters
BATCH_SIZE = 64
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 100
#———————————————————
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #tell the code to use the GPU if available

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(     #creat the iterators(can automatic put the data to the NN)
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device,
    sort_key=lambda x: len(x.text))

# HAT_RD model
class HAT_RD(nn.Module):
    def __init__(self,word_bilstm,post_bilstm,device):
        super().__init__()
        self.word_bilstm=word_bilstm
        self.post_bilstm=post_bilstm
        self.device = device
        
    def forward(self, batch_data):
        #batch_data.size()=[event len, batch]
        post_batch=torch.cuda.IntTensor([batch_data.size()[1]])
        post_len=torch.cuda.IntTensor([0])
        input_data=torch.cuda.IntTensor()
        # '''Important: You need to change the second latitude value according to the number of posts in the event, and the value is: 2*post number---256'''
        post_input=torch.ones((post_batch,256,512)).to(device)
        # Reorganize post according to event
        for batch_num in range(post_batch):   
            rdata=batch_data[:,batch_num]
            # Get word_batch
            word_batch=torch.cuda.LongTensor([len(rdata[rdata==2])])
            text=torch.cuda.LongTensor()
            textall=torch.cuda.LongTensor()
            # Get the data latitude value of the post [number of posts, the longest post length] = [word_batch, max]
            max=torch.cuda.LongTensor([0]) #max Record the length of the longest post under the event
            for i in rdata:
                if i!=2:
                    temple=torch.cuda.LongTensor([i.item()])
                    text=torch.cat((text,temple),0)
                else:
                    #textall=torch.cat([textall,text],0)
                    if max<torch.cuda.LongTensor([len(text)]):
                        max=torch.cuda.LongTensor([len(text)])
                    text=torch.cuda.LongTensor()
            
            inputs_cuda=torch.ones([word_batch,max]).to(device)# Initialize all for a while, used to create new word_post data
            # Reconstruct the data and fill in the data according to word_batch
            row_num=torch.cuda.LongTensor([0])
            clum_num=torch.cuda.LongTensor([0])
            for i in rdata:
                if i!=2:
                    inputs_cuda[row_num.item()][clum_num.item()]=i                    
                    clum_num=clum_num+1
                else:
                    row_num=row_num+1
                    clum_num=torch.cuda.LongTensor([0])
                    if row_num==word_batch:
                        break
            inputs_cuda=inputs_cuda.permute(1,0).type(torch.long)       #inputs [post len,batch] 

            word_hidden,word_result=word_bilstm(inputs_cuda)

            #word_hidden = [n layers * n directions, batch size, hid dim]=【2,nums of posts，256】
           # '''Dimensions need to be changed to pass HIDDEN_DIM with parameters'''
            word_hidden=word_hidden.view(-1,512)
            for i in range(word_hidden.size()[0]):
                for j in range(word_hidden.size()[1]):
                    post_input[batch_num][i][j]=word_hidden[i][j]
        
        #post_input=[event_len,batch,hid_dim]
        post_input=post_input.permute(1,0,2).to(device)
        result=post_bilstm(post_input)

        return result,word_result
    
word_bilstm=Word_BiLSTM(INPUT_DIM,EMBEDDING_DIM,HIDDEN_DIM,N_LAYERS,DROPOUT)
post_bilstm=Post_BiLSTM(HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)


pretrained_embeddings = TEXT.vocab.vectors
word_bilstm.embedding.weight.data.copy_(pretrained_embeddings)

model = HAT_RD(word_bilstm,post_bilstm,device).to(device)


event_optimizer = optim.Adam(model.parameters())
post_optimizer= optim.Adam(model.word_bilstm.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)


print('Start adversarial training!')
# training!
for epoch in range(N_EPOCHS):

    train_loss, train_acc = train_hat(model, train_iterator, event_optimizer, post_optimizer, criterion, 0.01)
    valid_loss, valid_acc, valid_recall_0, valid_recall_1, valid_precision_0, valid_precision_1, valid_f1_0, valid_f1_1 = evaluate(model, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02}')
    print(f'| Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'| Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
    print(f'| Label 0 | Val. Rec: {valid_recall_0*100:.2f}% | Val. Pre: {valid_precision_0*100:.2f}% | Val. F1: {valid_f1_0*100:.2f}%')
    print(f'| Label 1 | Val. Rec: {valid_recall_1*100:.2f}% | Val. Pre: {valid_precision_1*100:.2f}% | Val. F1: {valid_f1_1*100:.2f}%')
    print('='*100)




