from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self, learning_rate, weight_decay, epochs, gpu):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(500, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.max_epochs = epochs

        # indicate if the network module is created through training
        self.fitted = False

        #set criterion
        self.criterion = F.nll_loss

        # selection of device to use
        self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() and gpu is not None else "cpu")
        self.gpu = gpu
        if self.device == "cpu":
            self.gpu = None

    #FIXME eventualmente ricontrollare
    def init_net(self, m):
        #reset all parameters for Conv2d layer
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
            # m.weight.data.fill_(0.01)
            # m.bias.data.fill_(0.01)
        #reset all parameters for Linear layer
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.01)
            m.bias.data.fill_(0.01)

    def reset_parameters(self):
        self.apply(self.init_net)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def fit(self, train_loader):
        # reset parameters for all test
        self.reset_parameters()
        self.train()
        for epochs in range(self.max_epochs):
            # debug print
            # print('epochs: '+ epochs.__str__())
            for data in train_loader:
                x,y=data
                if self.gpu is not None:
                    x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self(x)
                train_loss = self.criterion(output, y)
                train_loss.backward()
                self.optimizer.step()
        self.fitted = True
        return train_loss

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
                    if self.gpu is not None:
                        x, y = x.to(self.device), y.to(self.device)

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