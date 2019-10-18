from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, learning_rate, weight_decay, epochs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.max_epochs = epochs

        # indicate if the network module is created through training
        self.fitted = False

        #set criterion
        self.criterion=F.nll_loss

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def fit(self, train_loader):
        self.train()
        for epochs in range(self.max_epochs):
            for data, target in train_loader:
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self(data)
                train_loss = self.criterion(output, target)
                train_loss.backward()
                self.optimizer.step()
        self.fitted = True

    def validation(self, validation_loader):
        if not self.fitted:
            exit(1)
        else:

            correct = 0
            total = 0
            loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for data in validation_loader:

                    # get some test images
                    x, y = data
                    # if self.gpu is not None:
                    #     images, labels = images.to(self.device), labels.to(self.device)

                    # images classes prediction
                    outputs = self(x)
                    _, predicted = torch.max(outputs.data, 1)

                    # loss update
                    loss += self.criterion(outputs, y).item()
                    num_batches += 1

                    # update numbers of total and correct predictions
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = correct / total
            loss /= num_batches
            return loss, accuracy


