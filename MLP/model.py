# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Res(nn.Module):
    def __init__(self, input_size):
        super(MLP_Res, self).__init__()
        self.input_size = input_size

        # self.fc1 = nn.Linear(self.input_size, self.input_size*4)
        self.fc1_l1 = nn.Linear(self.input_size, self.input_size+300)
        self.fc1_l2 = nn.Linear(self.input_size+300, self.input_size+500)
        self.fc1_rw = nn.Linear(self.input_size, self.input_size+500)
        self.fc1_ln = nn.LayerNorm(self.input_size+500)

        # self.fc2 = nn.Linear(self.input_size*4, self.input_size)
        self.fc2_l1 = nn.Linear(self.input_size+500, self.input_size)
        self.fc2_l2 = nn.Linear(self.input_size, self.input_size/2)
        self.fc2_rw = nn.Linear(self.input_size+500, self.input_size/2)
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
        del dout_ret
        del dout_res
        # predict = F.sigmoid(dout)
        predict = F.softmax(dout, dim=1)
        del dout

        return predict


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, self.input_size+300)
        self.fc2 = nn.Linear(self.input_size+300, self.input_size)
        self.fc3 = nn.Linear(self.input_size, self.input_size/2)
        self.fc4 = nn.Linear(self.input_size/2, self.input_size/4)
        self.fc5 = nn.Linear(self.input_size/4, 2)

    def forward(self, din):
        dout = F.relu(self.fc1(din), inplace=True)
        dout = F.relu(self.fc2(dout), inplace=True)
        dout = F.relu(self.fc3(dout), inplace=True)
        dout = F.relu(self.fc4(dout), inplace=True)
        dout = F.relu(self.fc5(dout), inplace=True)
        predict = F.softmax(dout, dim=1)
        del dout

        return predict


class MLP_1(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_1, self).__init__()
        self.fc1 = torch.nn.Linear(400, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 50)
        self.fc4 = torch.nn.Linear(50, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)


class MLP_2(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_2, self).__init__()
        self.fc1 = torch.nn.Linear(468, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 50)
        self.fc4 = torch.nn.Linear(50, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)


class MLP_3(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_3, self).__init__()
        self.fc1 = torch.nn.Linear(6100, 3000)
        self.fc2 = torch.nn.Linear(3000, 1000)
        self.fc3 = torch.nn.Linear(1000, 100)
        self.fc4 = torch.nn.Linear(100, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        # din = din.view(-1, 8000)
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)


class MLP_4(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_4, self).__init__()
        self.fc1 = torch.nn.Linear(6168, 3000)
        self.fc2 = torch.nn.Linear(3000, 1000)
        self.fc3 = torch.nn.Linear(1000, 100)
        self.fc4 = torch.nn.Linear(100, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        # din = din.view(-1, 8000)
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)


class MLP_5(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_5, self).__init__()
        self.fc1 = torch.nn.Linear(1600, 800)
        self.fc2 = torch.nn.Linear(800, 400)
        self.fc3 = torch.nn.Linear(400, 100)
        self.fc4 = torch.nn.Linear(100, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        # din = din.view(-1, 8000)
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)


class MLP_6(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP_6, self).__init__()
        self.fc1 = torch.nn.Linear(1668, 800)
        self.fc2 = torch.nn.Linear(800, 400)
        self.fc3 = torch.nn.Linear(400, 100)
        self.fc4 = torch.nn.Linear(100, 10)
        self.fc5 = torch.nn.Linear(10, 2)

    def forward(self, din):
        # din = din.view(-1, 8000)
        dout = F.leaky_relu(self.fc1(din), inplace=True)
        dout = F.leaky_relu(self.fc2(dout), inplace=True)
        dout = F.leaky_relu(self.fc3(dout), inplace=True)
        dout = F.leaky_relu(self.fc4(dout), inplace=True)
        return F.softmax(self.fc5(dout), dim=1)
