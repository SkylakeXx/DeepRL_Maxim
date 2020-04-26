import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 84, kernel_size=3, stride=2),
            nn.BatchNorm2d(84),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(84, 168, kernel_size=2, stride=1),
            nn.BatchNorm2d(168),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(168, 168, kernel_size=2),
            nn.BatchNorm2d(168),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)