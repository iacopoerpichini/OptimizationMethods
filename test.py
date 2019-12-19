from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from network import Net
import dataset_management




if __name__ == '__main__':
    train_loader, validation_loader, test_loader = dataset_management.getDataset(validation=True,dataset_name='mnist')

    model=Net(0.1,0.2,1)

    # train_loader, test_loader, validation_loader = dataset_menagement.getDataset(validation=Treu, dataset_name='cifar10')
    for epoch in range(1, 10):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.fit(train_loader)
