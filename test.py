from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from network import LeNet5
import dataset_menagement

batch_size = 64

model = LeNet5()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    train_loader, test_loader = dataset_menagement.getDataset(dataset_name='mnist')

    # train_loader, test_loader, validation_loader = dataset_menagement.getDataset(validation=Treu, dataset_name='cifar10')
    for epoch in range(1, 10):
        train(train_loader,epoch)
        test(test_loader)