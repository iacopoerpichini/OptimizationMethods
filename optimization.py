from __future__ import print_function
import dataset_management
from network import Net
import torch
import optunity
import csv

# param for experiment
max_epochs = 30
output_file = 'result.csv'
# gpu id
gpu = 0


def evaluate_BAY(learning_rate, weight_decay):
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model = Net(learning_rate, weight_decay, max_epochs,gpu).to(device)
    # here is possible to select MNIST of CIFAR10 dataset
    train_loader, validation_loader, test_loader = dataset_management.getDataset(validation=True, dataset_name='mnist')
    training_losses = model.fit(train_loader)
    validation_losses, validation_accuracy = model.validation(test_loader)
    best_val_loss = validation_losses

    # print("Accuracy Validation: " + str(validation_accuracy))
    # print('--------')
    # print(validation_losses,learning_rate,weight_decay)

    #print('weight:'+ validation_accuracy.__str__())

    #print('weight:'+ weight_decay.__str__())

    with open(output_file, 'a') as file:
        myCsvRow = 'iter,' + best_val_loss.__str__() + ',' + learning_rate.__str__() + ',' + weight_decay.__str__() + ',' + validation_accuracy.__str__() + '\n'
        file.write(myCsvRow)


    return -best_val_loss


