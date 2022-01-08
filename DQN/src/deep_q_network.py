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


class MLP(nn.Module):
    def __init__(self, input_num: int = 4, hidden_num: int = 64):
        super(MLP, self).__init__()
        print("---MLP{} init---".format(hidden_num))
        self.conv1 = nn.Sequential(nn.Linear(input_num, hidden_num), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(hidden_num, hidden_num), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(hidden_num, 1))

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