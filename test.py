from __future__ import print_function
from optimization import evaluate_BAY
from bayes_opt import BayesianOptimization
import optunity
import csv
from datetime import datetime


# param for experiment
output_file = 'result.csv'
num_evaluations = 25
init_points_BAY = 5

# hyperparameters domains
hyp_domains = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}

def bayesian():
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=hyp_domains)
    bay_opt.maximize(init_points=init_points_BAY, n_iter=num_evaluations-init_points_BAY)
    return '\nResults with Bayesian optimizer: ' + str(bay_opt.max) + '\n'

def quasiRandom():
    maximum=optunity.maximize(f=evaluate_BAY,num_evals=num_evaluations,solver_name='sobol',learning_rate=[0.0001,0.1], weight_decay=[0, 0.001])
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






