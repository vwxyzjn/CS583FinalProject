### File		: mnist.py
#  Author	    : David Grethlein
#  Organization : Drexel University
#  Date		    : May 21, 2019

## Note		    : RUN THIS SCRIPT USING PYTHON 3 NOT PYTHON 2!!!!!!!!!!!!!!
import argparse
import typing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable


BATCH_SIZE = 32

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def load_mnist_training_set(verbose : bool = False):
    X_train , y_train = loadlocal_mnist(
        images_path='./data/MNIST/raw/train-images-idx3-ubyte' , 
        labels_path='./data/MNIST/raw/train-labels-idx1-ubyte')
    
    train_class_labels = np.unique(y_train)
    train_class_dist = np.bincount(y_train)
    
    if verbose:
        print("\nTraining Data Shape: %s" % str(X_train.shape))
        print("Training Labels Shape: %s" % str(y_train.shape))
        print("Class Labels: %s" % train_class_labels)
        print("Class Distributions: %s" % train_class_dist)

    return tuple([X_train , y_train])


def load_mnist_test_set(verbose : bool = False):
    X_test , y_test = loadlocal_mnist(
        images_path='./data/MNIST/raw/t10k-images-idx3-ubyte' ,
        labels_path='./data/MNIST/raw/t10k-labels-idx1-ubyte')
    
    test_class_labels = np.unique(y_test)
    test_class_dist = np.bincount(y_test)
    
    if verbose:
        print("\nTest Data Shape: %s" % str(X_test.shape))
        print("Test Labels Shape: %s" % str(y_test.shape))
        print("Class Labels: %s" % test_class_labels)
        print("Class Distributions: %s" % test_class_dist)

    return tuple([X_test , y_test])

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def load_fashion_mnist_training_set(verbose : bool = False):
    X_train , y_train = loadlocal_mnist(
        images_path='./data/Fashion_MNIST/raw/train-images-idx3-ubyte' , 
        labels_path='./data/Fashion_MNIST/raw/train-labels-idx1-ubyte')
    
    train_class_labels = np.unique(y_train)
    train_class_dist = np.bincount(y_train)
    
    if verbose:
        print("\nTraining Data Shape: %s" % str(X_train.shape))
        print("Training Labels Shape: %s" % str(y_train.shape))
        print("Class Labels: %s" % train_class_labels)
        print("Class Distributions: %s" % train_class_dist)

    return tuple([X_train , y_train])


def load_fashion_mnist_test_set(verbose : bool = False):
    X_test , y_test = loadlocal_mnist(
        images_path='./data/Fashion_MNIST/raw/t10k-images-idx3-ubyte' ,
        labels_path='./data/Fashion_MNIST/raw/t10k-labels-idx1-ubyte')
    
    test_class_labels = np.unique(y_test)
    test_class_dist = np.bincount(y_test)
    
    if verbose:
        print("\nTest Data Shape: %s" % str(X_test.shape))
        print("Test Labels Shape: %s" % str(y_test.shape))
        print("Class Labels: %s" % test_class_labels)
        print("Class Distributions: %s" % test_class_dist)

    return tuple([X_test , y_test])

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def visualize(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, 
                    batch_idx*len(X_batch),
                    len(train_loader.dataset), 
                    100.*batch_idx / len(train_loader), 
                    loss.item(), 
                    float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


def evaluate(model):
    correct = 0 
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    # Loads the datasets from file
    X_train , y_train = load_mnist_training_set()
    X_test , y_test = load_mnist_test_set()

    X_train , y_train = load_fashion_mnist_training_set()
    X_test , y_test = load_fashion_mnist_test_set()

    # Converts the dataset into PyTorch LongTensor(s)
    torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # Reshape flattened images (784,) to square (28,28) 
    torch_X_train = torch_X_train.view(-1, 1,28,28).float()
    torch_X_test = torch_X_test.view(-1,1,28,28).float()
    #X_train = X_train.reshape(X_train.shape[0] , 28 , 28 , 1) 
    #X_test = X_test.reshape(X_test.shape[0] , 28 , 28 , 1)

    # Create tensor datasets linking samples to labels
    train = torch.utils.data.TensorDataset(torch_X_train , torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test , torch_y_test)

    train_loader = torch.utils.data.DataLoader(train , batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test , batch_size=BATCH_SIZE, shuffle=False)


    cnn = CNN()
    print(cnn)

    fit(cnn, train_loader)
    evaluate(cnn)
    
    # utilities scripts
    plt.imshow(torch_X_train[2][0])
