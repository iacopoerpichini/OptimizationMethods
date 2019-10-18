from __future__ import print_function
import dataset_management
from network import Net
import torch
import csv

from bayes_opt import BayesianOptimization

# output files
log_file = "log.txt"
csv_evaluations_file = "evaluations.csv"

max_epochs = 70
num_evaluations = 25
init_points_BAY = 5

# hyperparameters domains
hyp_domains = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}

# gpu id
gpu = 0


def evaluate_BAY(learning_rate, weight_decay):

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model = Net(learning_rate, weight_decay, max_epochs).to(device)
    train_loader, validation_loader, test_loader = dataset_management.getDataset(validation=True, dataset_name='mnist')
    training_losses,  training_accuracies = model.fit(train_loader)

    validation_losses, validation_accuracy = model.validation(test_loader)
    best_val_loss = min(validation_losses)


    return -best_val_loss

if __name__ == '__main__':
    hyp_opt = "BAY"
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=hyp_domains)
    bay_opt.maximize(init_points=init_points_BAY, n_iter=num_evaluations-init_points_BAY)
    print("Results with Bayesian optimizer: " + str(bay_opt.max) + "\n")