from __future__ import print_function
import dataset_management
from network import Net
import torch
import optunity
import csv


max_epochs = 50

# gpu id
gpu = 0


def evaluate_BAY(learning_rate, weight_decay):

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model = Net(learning_rate, weight_decay, max_epochs,gpu).to(device)
    train_loader, validation_loader, test_loader = dataset_management.getDataset(validation=True, dataset_name='mnist')
    training_losses = model.fit(train_loader)
    validation_losses, validation_accuracy = model.validation(test_loader)
    best_val_loss = validation_losses


    return -best_val_loss


