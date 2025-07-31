'''
In this file, we define the fixed parameters that we use for the
SQA hyperparameters tuning analysis
'''

# SQA parameters
n_spins = 40
n_shots = 32
delta_t = 0.01

# Bayesian optimization parameters
n_init = 60  # Number of initial random points for the Bayesian optimization
n_iters = 120 # Number of iterations for the Bayesian optimization

# Select min and max values of each hyperparameters: [beta, annealing_time, gamma_0, n_trotters]
min_vals = [0.001, 1.0, 0.1, 15]
max_vals = [100.0, 1000.0, 1000.0, 60]


    

