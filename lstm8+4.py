#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:13:39 2018

@author: zhukai
"""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
#import torch.utils.data as Data
from torch import nn
from torch.autograd import Variable

## load data
path = './csv'
files = os.listdir(path)
datas = []
counts = []
oxyHbs = []
deoxyHbs = []
chazhis = []
targets = []
oxy_des = []

for file in files:
    tmp = np.loadtxt("./csv/"+file ,dtype = np.str, delimiter = ",")
    data = tmp[36:,:].astype(np.float)
    time = data[:,0]
    num = len(time)
    count = data[0:num,3]
    target = np.mod(count,2)
    oxyHb = data[:,4:70:3]
    deoxyHb = data[:,5:71:3]
    oxy_de = np.concatenate((oxyHb,deoxyHb), axis=1)
    chazhi = oxyHb - deoxyHb
    datas.append(data)
    counts.append(count)
    targets.append(target)
    oxyHbs.append(oxyHb)
    deoxyHbs.append(deoxyHb)
    oxy_des.append(oxy_de)
    chazhis.append(chazhi)
    

TIME_STEP = 10      # rnn time step
INPUT_SIZE = 44      # rnn input size
LR = 0.02           # learning rate    
class mylstm(nn.Module):
    def __init__(self):
        super(mylstm, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
                )
        
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
 
net = mylstm()
#torch.save(net,'./lstm_chazhi.pth')
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
criterion = nn.MSELoss()

for i in range(22):
    ends1 = np.where(counts[i] == 16)
    end1 = ends1[0][0]

    oxy_des1 = oxy_des[i]
    input1 = oxy_des1[0:end1+100,:]
    target1 = targets[0]
    out = target1[0:end1+100]
    input1 = torch.from_numpy(input1[np.newaxis,:,:])
    input1 = input1.float()
    input1 = Variable(input1, requires_grad=False)
    target1 = torch.from_numpy(out[np.newaxis,:,np.newaxis])
    target1 = target1.float()
    target1 = Variable(target1, requires_grad=False)

    test_input = oxy_des1[end1+100:,:]
    count1 = counts[i]
    test_out = count1[end1+100:]
    test_input = torch.from_numpy(test_input[np.newaxis,:,:])
    test_input = test_input.float()
    test_input = Variable(test_input, requires_grad=False)
    for ii in range(100):
#        h_state position can change to the out of circulation
        h_state = None
        print('STEP: ', ii)
        optimizer.zero_grad()
        out, h_state = net(input1, h_state)
        h_state = [Variable(h_state[0].data),Variable(h_state[1].data)]
        loss = criterion(out, target1)
        print('loss:', loss.data.numpy()[0])
        loss.backward()
        optimizer.step()
    
#    torch.save(net,'./lstm8+4.pth') 
#    plt.figure(1)
#    plt.plot(out.data.numpy()[0][0])
    out1, h = net(test_input,None)
    plt.figure(i)
    plt.plot(out1.data.numpy()[0])
    for i in range(17,25):
        mark = i
        a = np.where(test_out == mark)
        b = a[0][0]
        plt.axvline(b)
#    filename = "./figure/" + "8+4_"+str(i+1) + ".jpg"
#    plt.savefig(filename)


