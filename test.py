from __future__ import print_function
from bayes_opt import BayesianOptimization
from datetime import datetime
import dataset_management
from network import Net
import torch
import optunity

# param for experiment
output_file = 'result.csv'
evaluations = 25
init_points = 5
max_epochs = 50
dataset_name= 'cifar10' # mnist
# gpu id for colab or gpu on pc
gpu = 0

# hyperparameters domains
hyperparameters = {'learning_rate': (0.0001, 0.1), 'weight_decay': (0, 0.001)}


def evaluate(learning_rate, weight_decay):
    device = torch.device('cuda:' + gpu.__str__() if torch.cuda.is_available() else 'cpu')
    model = Net(learning_rate, weight_decay, max_epochs, gpu, dataset_name=dataset_name).to(device)
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
        my_csv_row = 'iter,' + best_val_loss.__str__() + ',' + learning_rate.__str__() + ',' + weight_decay.__str__() + ',' + validation_accuracy.__str__() + '\n'
        file.write(my_csv_row)

    return -best_val_loss


def bayesian():
    bayesian = BayesianOptimization(f=evaluate, pbounds=hyperparameters)
    bayesian.maximize(init_points=init_points, n_iter=evaluations - init_points)
    return '\nResults with Bayesian optimizer: ' + str(bayesian.max) + '\n'

def quasi_random():
    quasi_random=optunity.maximize(f=evaluate, num_evals=evaluations, solver_name='sobol', learning_rate=[0.0001, 0.1], weight_decay=[0, 0.001])
    return '\nResult with quasiRandom optimizer: ' + str(quasi_random) + '\n'

if __name__ == '__main__':

    now = datetime.now()
    with open(output_file, 'a', newline='') as file:
        file.write('\nBayesian ' + now.isoformat() + '\n')
        file.write('\nIter, Loss, Learning Rate, Weight Decay, Accuracy Validation\n')

    result_bay = bayesian()
    print(result_bay)

    with open(output_file, 'a') as file:
        file.write(result_bay + '\n')

    print('------------')

    now = datetime.now()
    with open(output_file, 'a') as file:
        file.write('QuasiRandom ' + now.isoformat() + '\n')
        file.write('\nIter, Loss, Learning Rate, Weight Decay, Accuracy Validation\n')


    result_qr = quasi_random()
    print(result_qr)

    with open(output_file, 'a') as file:
        file.write(result_qr+'\n')






