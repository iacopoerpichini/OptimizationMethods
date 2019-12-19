from __future__ import print_function
from optimization import evaluate_BAY
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from network import Net
import dataset_management
from bayes_opt import BayesianOptimization
import optunity



# output files
log_file = "log.txt"
csv_evaluations_file = "evaluations.csv"
num_evaluations = 25
init_points_BAY = 5

# hyperparameters domains
hyp_domains = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}

def bayesian():
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=hyp_domains)
    bay_opt.maximize(init_points=init_points_BAY, n_iter=num_evaluations-init_points_BAY)
    print("Results with Bayesian optimizer: " + str(bay_opt.max) + "\n")

def quasiRandom():
    maximum=optunity.maximize(f=evaluate_BAY,num_evals=num_evaluations,solver_name='sobol',learning_rate=[0.0001,0.1], weight_decay=[0, 0.001])
    print("Result with quasiRandom optimizer: "+ str(maximum)+ "\n")

if __name__ == '__main__':
    quasiRandom() #or quasiRandom()





