from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict


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
                output = self.Net(data)
                train_loss = F.nll_loss(output, target)
                train_loss.backward()
                self.optimizer.step()
        self.fitted = True

    def validation(self, validation_loader):
        if not self.fitted:
            exit(1)
        else:

            validation_losses=[]
            validation_accuracies=[]
            validation_loss = 0.0
            validation_num_minibatches = 0
            validation_correct_predictions = 0
            validation_examples = 0


            for data, target in validation_loader:
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.Net(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                # validation_loss
                validation_loss += loss.item()
                validation_num_minibatches += 1

                # calculate correct predictions for accuracy
                _, predicted = torch.max(output.data, 1)
                validation_examples += data.size(0)
                validation_correct_predictions += (predicted == target).sum().item()


        validation_loss /= validation_num_minibatches
        validation_losses.append(validation_loss)
        validation_accuracy = validation_correct_predictions / validation_examples
        validation_accuracies.append(validation_accuracy)

        return validation_accuracies,validation_losses


