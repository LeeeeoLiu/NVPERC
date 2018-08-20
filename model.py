# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size

        # self.fc1 = nn.Linear(self.input_size, self.input_size*4)
        self.fc1_l1 = nn.Linear(self.input_size, self.input_size*2)
        self.fc1_l2 = nn.Linear(self.input_size*2, self.input_size*4)
        self.fc1_rw = nn.Linear(self.input_size, self.input_size*4)
        self.fc1_ln = nn.LayerNorm(self.input_size*4)

        # self.fc2 = nn.Linear(self.input_size*4, self.input_size)
        self.fc2_l1 = nn.Linear(self.input_size*4, self.input_size)
        self.fc2_l2 = nn.Linear(self.input_size, self.input_size/2)
        self.fc2_rw = nn.Linear(self.input_size*4, self.input_size/2)
        self.fc2_ln = nn.LayerNorm(self.input_size/2)

        # self.fc3 = nn.Linear(self.input_size, self.input_size/4)
        self.fc3_l1 = nn.Linear(self.input_size/2, self.input_size/4)
        self.fc3_l2 = nn.Linear(self.input_size/4, 2)
        self.fc3_rw = nn.Linear(self.input_size/2, 2)
        self.fc3_ln = nn.LayerNorm(2)


    def forward(self, din):

        dout_ret = self.fc1_l2(F.leaky_relu(self.fc1_l1(din), inplace=True))
        dout_res = self.fc1_rw(din)
        din = self.fc1_ln(dout_ret+dout_res)

        dout_ret = self.fc2_l2(F.leaky_relu(self.fc2_l1(din), inplace=True))
        dout_res = self.fc2_rw(din)
        din = self.fc2_ln(dout_ret+dout_res)

        dout_ret = self.fc3_l2(F.leaky_relu(self.fc3_l1(din), inplace=True))
        dout_res = self.fc3_rw(din)
        dout = self.fc3_ln(dout_ret+dout_res)

        return F.softmax(self.fc5(dout))
