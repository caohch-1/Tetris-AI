import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)

        self.fc1 = nn.Sequential(nn.Linear(20 * 12 * 2, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 20*10 -> 10*16*6
        x = F.relu(self.conv2(x))  # 10*16*6 -> 20*12*2
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class DQN64(nn.Module):
    def __init__(self):
        super(DQN64, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class DQN128(nn.Module):
    def __init__(self):
        super(DQN128, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)

        return x


class DQN256(nn.Module):
    def __init__(self):
        super(DQN256, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.out(x)

        return x
