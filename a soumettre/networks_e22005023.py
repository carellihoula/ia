import itertools
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import h5py
import json
import copy
import time
from datetime import datetime

#-------CREATION D'UN MODELE SIMPLECNN ET CombinedCNNLSTM QUI COMBINE CNN ET LSTM SE TROUVE TOUT EN BAS----------------

def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, seq):
        seq = torch.flatten(seq, start_dim=1)
        x = F.dropout(F.relu(self.lin1(seq)), p=0.1)
        x = F.dropout(F.relu(self.lin2(x)), p=0.1)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer, scheduler):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        train_loss_list = []
        dev_loss_list = []
        
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            scheduler.step()
            loss_batch /= nb_batch
            train_loss_list.append(loss_batch / nb_batch)

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            dev_loss = loss_batch
            dev_loss_list.append(dev_loss)
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        self.plot_learning_curve(train_acc_list, dev_acc_list, train_loss_list, dev_loss_list)
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
        # Now calculate recall and F1-score
        """report = classification_report(all_targets, all_predicts, output_dict=True)
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print(f"Recall: {recall}, F1 Score: {f1_score}")  """                  
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
    #Ajout d'une methode pour Tracer la courbe d'apprentissage 
    def plot_learning_curve(self, train_performance, dev_performance, train_loss, dev_loss):
        epochs = range(1, len(train_performance) + 1)

        # Plotting performance
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_performance, label='Train')
        plt.plot(epochs, dev_performance, label='Dev')
        plt.title('Training and Dev Performance')
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.legend()

        # Plotting loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Train')
        plt.plot(epochs, dev_loss, label='Dev')
        plt.title('Training and Dev Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim) #ajout
        #1st option: using hidden states
        # self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        #2nd option: using output seauence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1) #pour reduire du surapprentissage

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out = self.dropout(lstm_out)

        #1st option: using hidden states as below
        # outp = self.hidden2output(torch.cat((hn,cn),-1))
        
        #2nd option: using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
            #outp = F.softmax(outp, dim=1).squeeze()

        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer,scheduler):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_loss_list = []
        dev_loss_list = []
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            loss_batch /= nb_batch
            scheduler.step()
            train_loss_list.append(loss_batch / nb_batch)

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            dev_loss = loss_batch
            dev_loss_list.append(dev_loss)
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        self.plot_learning_curve(train_acc_list, dev_acc_list, train_loss_list, dev_loss_list)
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))

            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        # Now calculate recall and F1-score
        """report = classification_report(all_targets, all_predicts, output_dict=True)
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        print(f"Recall: {recall}, F1 Score: {f1_score}")  """

        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
    #Pour Tracer la courbe d'apprentissage 
    def plot_learning_curve(self, train_performance, dev_performance, train_loss, dev_loss):
        epochs = range(1, len(train_performance) + 1)

        # Plotting performance
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_performance, label='Train')
        plt.plot(epochs, dev_performance, label='Dev')
        plt.title('Training and Dev Performance')
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.legend()

        # Plotting loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Train')
        plt.plot(epochs, dev_loss, label='Dev')
        plt.title('Training and Dev Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
#-----------------------------------CREATION D'UN MODELE QUI COMBINE CNN ET LSTM----------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512) # Adjust the input size
        self.fc2 = nn.Linear(512, 10) # 10 classes for example

    def forward(self, x):
        # Applying layers and activations
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Add these methods to the SimpleCNN class

    def train_all(self, train_loader, dev_loader, num_epochs, device, optimizer):
        self.to(device)
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            total_loss = 0
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            # Evaluate on the development set
            dev_accuracy = self.evaluate(dev_loader, device)
            print(f"Epoch {epoch+1} - Dev Accuracy: {dev_accuracy}")

    def evaluate(self, test_loader, device):
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total
        return accuracy


class CombinedCNNLSTM(nn.Module):
    def __init__(self, conf):
        super(CombinedCNNLSTM, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM layers
        self.lstm_input_size = 64 * 4 * 4  # Example input size after CNN and pooling
        self.lstm = nn.LSTM(self.lstm_input_size, conf["LSTM_conf"]["hidden_dim"], 
                            batch_first=True, 
                            num_layers=conf["LSTM_conf"]["num_layers"])
        # Output layer
        self.fc = nn.Linear(conf["LSTM_conf"]["hidden_dim"], conf["board_size"]*conf["board_size"])

    def forward(self, x):
        # CNN part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for LSTM
        x = x.unsqueeze(1)  # Add sequence dimension
        # LSTM part
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the last LSTM output
        # Output layer
        x = self.fc(lstm_out)
        return torch.sigmoid(x)

    def train_all(self, train_loader, dev_loader, num_epochs, device, optimizer):
        self.to(device)
        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            total_loss = 0
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            # Evaluate on the development set
            dev_accuracy = self.evaluate(dev_loader, device)
            print(f"Epoch {epoch+1} - Dev Accuracy: {dev_accuracy}")

    def evaluate(self, test_loader, device):
        self.eval()
        all_predicts = []
        all_targets = []
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                all_predicts.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Performance Metrics
        accuracy = np.mean(np.array(all_predicts) == np.array(all_targets))
        print(f"Accuracy: {accuracy}")
        
        # Additional Metrics
        print(classification_report(all_targets, all_predicts, digits=4))

        # Confusion Matrix
        conf_matrix = confusion_matrix(all_targets, all_predicts)
        self.plot_confusion_matrix(conf_matrix)

        return accuracy

    def plot_confusion_matrix(self, cm,all_targets):
        plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(all_targets)))  # Adjust number of classes
        plt.xticks(tick_marks, range(len(set(all_targets))))
        plt.yticks(tick_marks, range(len(set(all_targets))))

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

          

