import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Quasi import Quasi


## THIS FILE IS FOR SAVING NET CLASSES FOR TORCH LOADING

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ReLULayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = ReLULayer(128, 16)
        self.fc5 = nn.Linear(16, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = ReLULayer(128, 8)
        self.fc5 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(10, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class TitanicQTNet(nn.Module):
    def __init__(self):
        super(TitanicQTNet, self).__init__()
        self.q1 = Quasi(9, 16)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.q1(x)
        x = F.tanh(self.fc1(x))
        return x


class TitanicTQNet(nn.Module):
    def __init__(self):
        super(TitanicTQNet, self).__init__()
        self.fc1 = nn.Linear(9, 2)
        self.q1 = Quasi(2, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x


class TitanicQTQNet(nn.Module):
    def __init__(self):
        super(TitanicQTQNet, self).__init__()
        self.q1 = Quasi(9, 16)
        self.fc1 = nn.Linear(16, 8)
        self.q2 = Quasi(8, 1)

    def forward(self, x):
        x = self.q1(x)
        x = F.tanh(self.fc1(x))
        x = self.q2(x)
        return x


class SpiralsTQNet(nn.Module):
    def __init__(self):
        super(SpiralsTQNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.q1 = Quasi(8, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x

class SpiralsTQ2Net(nn.Module):
    def __init__(self):
        super(SpiralsTQ2Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.q1 = Quasi(16, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x


class SpiralsTQ3Net(nn.Module):
    def __init__(self):
        super(SpiralsTQ3Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.q1 = Quasi(20, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x


class SpiralsQTNet(nn.Module):
    def __init__(self):
        super(SpiralsQTNet, self).__init__()
        self.q1 = Quasi(2, 16)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.q1(x)
        x = F.tanh(self.fc1(x))
        return x

class SpiralsTQTQNet(nn.Module):
    def __init__(self):
        super(SpiralsTQTQNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.q1 = Quasi(5, 8)
        self.fc2 = nn.Linear(8, 5)
        self.q2 = Quasi(5,1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        x = F.tanh(self.fc2(x))
        x = self.q2(x)
        return x


class SpiralsTQTQ2Net(nn.Module):
    def __init__(self):
        super(SpiralsTQTQ2Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.q1 = Quasi(10,96)
        self.fc2 = nn.Linear(96,5)
        self.q2 = Quasi(5,1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        x = F.tanh(self.fc2(x))
        x = self.q2(x)
        return x

class SpiralsTTTQ3Net(nn.Module):
    def __init__(self):
        super(SpiralsTTTQ3Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.q1 = Quasi(8, 1)


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.q1(x)
        return x

class SpiralsMLPNet(nn.Module):
    def __init__(self):
        super(SpiralsMLPNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

class Spirals2MLPNet(nn.Module):
    def __init__(self):
        super(Spirals2MLPNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


class Spirals3MLPNet(nn.Module):
    def __init__(self):
        super(Spirals3MLPNet, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

class Parity4Net(torch.nn.Module):
    def __init__(self, h):
        super(Parity4Net, self).__init__()
        self.fc1 = nn.Linear(4, h)
        self.q1 = Quasi(h, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.q1(x)
        return x







####### TESTING NETS #######

class Plus(nn.Module):
    def __init__(self):
        super(Plus, self).__init__()

    def forward(self, x):
        return x + torch.ones(x.size()).type_as(x)

class Minus(nn.Module):
    def __init__(self):
        super(Minus, self).__init__()

    def forward(self, x):
        return x - torch.ones(x.size()).type_as(x)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()

        self.fc1 = Plus()

    def forward(self, x):
        return self.fc1(x)


