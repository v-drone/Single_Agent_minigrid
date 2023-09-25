from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_actions):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=512)

        # Output layer
        self.output = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = nn.ReLU()(self.fc(x))
        return self.output(x)
