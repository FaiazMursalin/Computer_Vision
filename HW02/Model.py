import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        '''
        defining cnn model with 3 layers
        '''
        super(CNNModel, self).__init__()
        '''First Layer:
        - Convolutional Layer with 16 filters, kernel size of 3x3, stride of 1, and padding of 1.
        - ReLU Activation Layer.
        - MaxPooling Layer with kernel size of 2x2 and stride of 2.'''
        self.layer1 = nn.Sequential(
            # 3 in channels because rgb image the out channel will be the filter and that will
            # be the in channel for the second layer
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''Second Layer:
        - Convolutional Layer with 32 filters, kernel size of 3x3, stride of 1, and padding of 1.
        - ReLU Activation Layer.
        - MaxPooling Layer with kernel size of 2x2 and stride of 2.'''
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''Third Layer:
        - Convolutional Layer with 64 filters, kernel size of 3x3, stride of 1, and padding of 1.
        - ReLU Activation Layer.
        - MaxPooling Layer with kernel size of 2x2 and stride of 2.'''
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''
        fully connected layer,
        for cifar 10 the images are of 32x32x3 so after 3 layers of max pooling 
        32->16(for first layer)->8(for second layer)->4(for third layer)
        '''
        # adding a larger intermediate layer
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            # dropout with 50% probability to avoid overfitting for 0.1 lr
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
        # without droput
        # self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        '''
        forward propagation
        :param x:
        :return:
        '''
        # first layer
        x = self.layer1(x)
        # second layer
        x = self.layer2(x)
        # third layer
        x = self.layer3(x)
        # flatten the output from th third layer
        # (batchsize, num of feature)
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = self.fc1(x)
        # return x
        return x


# utility function to load the model
def load_model(path):
    model = CNNModel()
    model.load_state_dict(torch.load(path))
    return model
