from __future__ import print_function
from bayes_opt import BayesianOptimization
from datetime import datetime
import dataset_management
from network import Net
import torch
import optunity

# param for experiment
output_file = 'result.csv'
num_evaluations = 1
init_points = 0
max_epochs = 1
dataset_name= 'cifar10' # mnist
# gpu id
gpu = 0

# hyperparameters domains
hyperparameters = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}


def evaluate(learning_rate, weight_decay):
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model = Net(learning_rate, weight_decay, max_epochs,gpu,dataset_name=dataset_name).to(device)
    # here is possible to select MNIST of CIFAR10 dataset
    train_loader, validation_loader, test_loader = dataset_management.getDataset(validation=True, dataset_name=dataset_name)
    training_losses = model.fit(train_loader)
    validation_losses, validation_accuracy = model.validation(test_loader)
    best_val_loss = validation_losses

    # print("Accuracy Validation: " + str(validation_accuracy))
    # print('--------')
    # print('Learning rate, weight decay')
    # print(learning_rate,weight_decay)

    # Save results in csv
    with open(output_file, 'a') as file:
        myCsvRow = 'iter,' + best_val_loss.__str__() + ',' + learning_rate.__str__() + ',' + weight_decay.__str__() + ',' + validation_accuracy.__str__() + '\n'
        file.write(myCsvRow)


    return -best_val_loss


def bayesian():
    bay_opt = BayesianOptimization(f=evaluate, pbounds=hyperparameters)
    bay_opt.maximize(init_points=init_points, n_iter=num_evaluations - init_points)
    return '\nResults with Bayesian optimizer: ' + str(bay_opt.max) + '\n'

def quasiRandom():
    maximum=optunity.maximize(f=evaluate, num_evals=num_evaluations, solver_name='sobol', learning_rate=[0.0001, 0.1], weight_decay=[0, 0.001])
    return '\nResult with quasiRandom optimizer: ' + str(maximum) + '\n'

if __name__ == '__main__':

    dateTimeObj = datetime.now()
    with open(output_file, 'a', newline='') as file:
        file.write('\nBayesian ' + dateTimeObj.isoformat() + '\n')
        file.write('\nIter, Loss, Learning Rate, Weight Decay, Accuracy Validation\n')

    result = bayesian()
    print(result)

    with open(output_file, 'a') as file:
        myCsvRow = result+'\n'
        file.write(myCsvRow)

    print('------------')

    dateTimeObj = datetime.now()
    with open(output_file, 'a') as file:
        file.write('QuasiRandom ' + dateTimeObj.isoformat() + '\n')
        file.write('\nIter, Loss, Learning Rate, Weight Decay, Accuracy Validation\n')


    result = quasiRandom()
    print(result)

    with open(output_file, 'a') as file:
        myCsvRow = result
        file.write(myCsvRow)






