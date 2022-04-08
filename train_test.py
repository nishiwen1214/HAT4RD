import torch
import torch.nn as nn
import numpy as np
from AT import *
from model import *
        
def train_hat(model, iterator, event_optimizer,post_optimizer, criterion,a):
    
    hat = AT(model)
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    best_acc = 0.
    
    for batch in iterator:
        
        event_optimizer.zero_grad()
        post_optimizer.zero_grad()
        
        event_predictions,post_predictions = model(batch.text)
        event_predictions=event_predictions.squeeze(1)
        if len(post_predictions.size())!=1:
            post_predictions=post_predictions.squeeze(1)
            
        if batch.rumour[-1]==1:
            post_rumour=torch.ones(post_predictions.size()).to(device)
        else:
            post_rumour=torch.zeros(post_predictions.size()).to(device)
 
        post_loss = criterion(post_predictions, post_rumour)
        event_loss = criterion(event_predictions, batch.rumour)
        #print("post_loss",post_loss)
        #print("event_loss",event_loss)
        acc,_,_,_ = binary_accuracy(event_predictions, batch.rumour)
        
        #para = list(model.parameters())
        
        loss=a*post_loss+(1-a)*event_loss

        loss.backward(retain_graph=True)  # Calculate the original gradient ，loss—total
        hat.perturb(1, 'word_bilstm.embedding.weight') # Add adversarial perturbation to post-level embedding
        event_loss.backward(retain_graph=True)  # Calculate the original gradient ，loss-e
        hat.perturb(0.3, 'word_bilstm.fc.bias') # # Add adversarial perturbation to event-level embedding
        
        
        event_predictions,post_predictions = model(batch.text)
        event_predictions=event_predictions.squeeze(1)
  
        if len(post_predictions.size())!=1:
            post_predictions=post_predictions.squeeze(1)
            
        if batch.rumour[-1]==1:
            post_rumour=torch.ones(post_predictions.size()).to(device)
        else:
            post_rumour=torch.zeros(post_predictions.size()).to(device)
 
        post_loss = criterion(post_predictions, post_rumour)
        event_loss = criterion(event_predictions, batch.rumour)
        #print("post_loss",post_loss)
        #print("event_loss",event_loss)
        acc,_,_,_ = binary_accuracy(event_predictions, batch.rumour)
        
        #para = list(model.parameters())
        event_optimizer.zero_grad()
        
        loss+= a*post_loss+(1-a)*event_loss
        hat.restore('word_bilstm.embedding.weight') # Restore post-level embedding parameters
        hat.restore('word_bilstm.fc.bias') # Restore event-level  embedding parameters
        
        loss.backward()
        event_optimizer.step()
        
        epoch_loss += event_loss.item()
        epoch_acc += acc.item()
#         if acc > best_acc:
#             best_acc = acc
#             torch.save(model.state_dict(), 'HAT-18-DA.pt')
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall_0 = 0
    epoch_precision_0 = 0
    epoch_f1_0 = 0
    epoch_recall_1 = 0
    epoch_precision_1 = 0
    epoch_f1_1 = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            #predictions = model(batch.text).squeeze(1)
            event_predictions,post_predictions = model(batch.text)
            event_predictions=event_predictions.squeeze(1)
            loss = criterion(event_predictions, batch.rumour) 
            
            #loss = criterion(predictions, batch.rumour)        
                     
            acc, recall, precision, f1 = binary_accuracy(event_predictions, batch.rumour)

            epoch_loss += loss.item() / len(iterator)
            epoch_acc += acc.item() / len(iterator)
            epoch_recall_0 += recall['0'].item() / len(iterator)
            epoch_recall_1 += recall['1'].item() / len(iterator)
            epoch_precision_0 += precision['0'].item() / len(iterator)
            epoch_precision_1 += precision['1'].item() / len(iterator)
            epoch_f1_0 += f1['0'].item() / len(iterator)
            epoch_f1_1 += f1['1'].item() / len(iterator)

    return epoch_loss, epoch_acc, epoch_recall_0, epoch_recall_1, epoch_precision_0, epoch_precision_1, epoch_f1_0, epoch_f1_1

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)

    
    predicted_0 = (rounded_preds == 0)
    predicted_1 = (rounded_preds == 1)

    tp_0 = ((predicted_0) & (y == 0)).float()
    fp_0 = ((predicted_0) & (y == 1)).float()
    tn_0 = ((predicted_1) & (y == 1)).float()
    fn_0 = ((predicted_1) & (y == 0)).float()
    precision_0 = tp_0.sum() / (tp_0.sum() + fp_0.sum())
    recall_0 = tp_0.sum() / (tp_0.sum() + fn_0.sum())
    f1_0 = 2 * ((precision_0 * recall_0) / (precision_0 + recall_0))
    
    tp_1 = ((predicted_1) & (y == 1)).float()
    fp_1 = ((predicted_1) & (y == 0)).float()
    tn_1 = ((predicted_0) & (y == 0)).float()
    fn_1 = ((predicted_0) & (y == 1)).float()
    precision_1 = tp_1.sum() / (tp_1.sum() + fp_1.sum())
    recall_1 = tp_1.sum() / (tp_1.sum() + fn_1.sum())
    f1_1 = 2 * ((precision_1 * recall_1) / (precision_1 + recall_1))
    
    precision = {'0' : precision_0, '1' : precision_1}
    recall = {'0' : recall_0, '1' : recall_1}
    f1 = {'0' : f1_0, '1' : f1_1}
    
    
    #print(correct_0)
    #print(correct_1)
    
    
    #tp = ((rounded_preds == 1) & (rounded_preds == y)).float()
    #fp = ((rounded_preds == 1) & (y == 0)).float()
    #tn = ((rounded_preds == y) & (rounded_preds == 0)).float()
    #fn = ((rounded_preds == 0) & (y == 1)).float()
    
    #precision  = tp.sum() / (tp.sum() + fp.sum())
    #recall  = tp.sum() / (tp.sum() + fn.sum())
    #f1 = 2 * ((recall * precision) / (recall + precision))
    
    return acc, precision, recall, f1
