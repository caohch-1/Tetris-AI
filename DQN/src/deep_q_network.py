import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_num: int = 4, hidden_num: int = 64):
        super(MLP, self).__init__()
        print("---MLP{}-{} init---".format(input_num, hidden_num))
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
