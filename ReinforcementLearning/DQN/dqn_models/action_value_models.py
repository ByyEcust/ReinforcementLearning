import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionValueModelFlappyBird(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        x = torch.zeros(state_shape).unsqueeze(0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        x = self.conv1(x)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        x = self.conv2(x)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        x = self.conv3(x).view(1, -1)

        self.fc = nn.Sequential(nn.Linear(x.shape[1], 256),
                                nn.ReLU(),
                                nn.Linear(256, action_shape))

    def forward(self, state):
        batch_size = state.shape[0]
        c1 = self.conv1(state)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        out = self.fc(c3.view(batch_size, -1))
        return out


