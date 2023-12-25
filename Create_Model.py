import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import scipy
import sys
import os
import glob
import time
import wfdb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

#Create Model


class ResBlock(nn.Module):
    #Residual Blocks
    def __init__(self, inp_feats):
        super(ResBlock, self).__init__()
        self.inp_feats=inp_feats
        self.conv1 = nn.Conv1d(inp_feats, 32, kernel_size=5, stride=1)      
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(48)


    def forward(self, x, reverse=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

#Feature Generator

class Feature(nn.Module):
    #Create Feature Generator (F)
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1=nn.Conv1d(1, 48, kernel_size=9, stride=1) 
        self.bn1 = nn.BatchNorm1d(48)
        self.conv2=nn.Conv1d(48, 48, kernel_size=9, stride=1) 
        self.bn2 = nn.BatchNorm1d(48)
        self.conv3=nn.Conv1d(48, 48, kernel_size=9, stride=1) 
        self.bn3 = nn.BatchNorm1d(48)

        self.G1 = ResBlock(1)
        self.G2 = ResBlock(48)
        self.G3 = ResBlock(48)
        self.mp=nn.MaxPool1d(2)

    def forward(self, x, reverse=False): 
        shortcut1= F.relu(self.bn1(self.conv1(x)))
        x = self.G1(x)
        x=torch.add(shortcut1, x) 
        x = self.mp(x)

        shortcut2= F.relu(self.bn2(self.conv2(x)))
        x = self.G2(x)
        x=torch.add(shortcut2, x)  

        x = self.mp(x)
        shortcut3= F.relu(self.bn3(self.conv3(x)))
        x = self.G3(x)
        x=torch.add(shortcut3, x)  

        x = self.mp(x)

        x = x.view(x.size(0), x.size(1)*x.size(2)) 
        return x


class Predictor(nn.Module):
    #Create Classifier (C)
    def __init__(self, num_classes, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(48*18, 100) 
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 10)
        self.bn2_fc = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(13, num_classes) #10 + 3(RR intervals, pre-RR and last eight pre-RR intervals)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)
        self.fc4 = nn.Softmax(dim=1)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, RRinterval, prevRR, prev_eightRR, reverse=False):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x=torch.cat([x,RRinterval,prevRR,prev_eightRR],dim=1) # (Add RR intervals, pre-RR and last eight pre-RR intervals)
    
        x_prev = self.fc3(x)
        x = self.fc4(x_prev)
        return x, x_prev  


def Generator(pixelda=False):   
        return Feature()

def Classifier(num_classes):
        return Predictor(num_classes)
