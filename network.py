from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


class Net(nn.Module):

    def __init__(self, learning_rate, weight_decay, epochs, gpu, dataset_name):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        if dataset_name == 'cifar10':
            self.conv1 = nn.Conv2d(3, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)
        if dataset_name == 'cifar10':
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
            print('epochs: '+ epochs.__str__())
            for batch in train_loader:
                data,target=batch
                if self.gpu is not None:
                    data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data)
                train_loss = self.criterion(output, target)
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
                for batch in validation_loader:
                    # get some test images
                    data, target = batch
                    if self.gpu is not None:
                        data, target = data.to(self.device), target.to(self.device)

                    # images classes prediction
                    outputs = self(data)
                    _, predicted = torch.max(outputs.data, 1)

                    # loss update
                    loss += self.criterion(outputs, target).item()
                    num_batches += 1

                    # update numbers of total and correct predictions
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = correct / total
            loss /= num_batches
            return loss, accuracy

# this main is only for see the network structure
if __name__ == '__main__':
    dataset_name = 'cifar10'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(learning_rate=0.0001, weight_decay=0.01, epochs=30, gpu=0,dataset_name=dataset_name).to(device)

    if dataset_name == 'mnist':
        summary(model, (1, 28, 28))
    if dataset_name == 'cifar10':
        summary(model, (3, 32, 32))