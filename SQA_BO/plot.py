'''
In this file, we generate plots for the results of the Bayesian optimization applied to the SQA problem.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##############################################################
################### Processing the tables ####################
##############################################################

# Load the results from the CSV file and convert to a DataFrame and then to numpy arrays
path = 'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/tables'

tables = {
    '5_15_30': pd.read_csv(f'{path}/BO_SQA_results_5_fixed_problem_bf_15_30.csv'),
    '10_15_30': pd.read_csv(f'{path}/BO_SQA_results_10_fixed_problem_bf_15_30.csv'),
    '10_30_60': pd.read_csv(f'{path}/BO_SQA_results_10_fixed_problem_bf_30_60.csv'),
    '15_30_60': pd.read_csv(f'{path}/BO_SQA_results_15_fixed_problem_bf_30_60.csv'),
    '15_50_100': pd.read_csv(f'{path}/BO_SQA_results_15_fixed_problem_bf_50_100.csv'),
    '17_30_60': pd.read_csv(f'{path}/BO_SQA_results_17_fixed_problem_bf_30_60.csv'),
    #'17_30_80': pd.read_csv(f'{path}/BO_SQA_results_17_fixed_problem_bf_30_80.csv'),
    '17_50_100': pd.read_csv(f'{path}/BO_SQA_results_17_fixed_problem_bf_50_100.csv'),
    '20_30_60': pd.read_csv(f'{path}/BO_SQA_results_20_fixed_problem_bf_30_60.csv'),
    '25_30_60': pd.read_csv(f'{path}/BO_SQA_results_25_fixed_problem_bf_30_60.csv'),
    '27_30_60': pd.read_csv(f'{path}/BO_SQA_results_27_fixed_problem_bf_30_60.csv')
}

# For each table, extract the average and standard deviation of the minimum energy
results = {}
for key, df in tables.items():
    n_spins = int(key.split('_')[0])
    n_init = int(key.split('_')[1])
    n_iters = int(key.split('_')[2])
    n_shots = df['n_shots'].values[0]
    delta_t = df['delta_t'].values[0]

    min_energy = df['min_energy'].values
    avg_min_energy = np.mean(min_energy)
    std_min_energy = np.std(min_energy)

    bo_time = df['execution_time'].values
    avg_BO_time = np.mean(bo_time)
    std_BO_time = np.std(bo_time)

    beta = df['opt_beta'].values
    avg_beta = np.mean(beta)
    std_beta = np.std(beta)
    max_beta = max(beta)
    min_beta = min(beta)

    annealing_time = df['opt_annealing_time'].values
    avg_annealing_time = np.mean(annealing_time)
    std_annealing_time = np.std(annealing_time)
    max_annealing_time = max(annealing_time)
    min_annealing_time = min(annealing_time)

    gamma_0 = df['opt_gamma_0'].values
    avg_gamma_0 = np.mean(gamma_0)
    std_gamma_0 = np.std(gamma_0)
    max_gamma_0 = max(gamma_0)
    min_gamma_0 = min(gamma_0)

    n_trotters = df['opt_n_trotters'].values
    avg_n_trotters = np.mean(n_trotters)
    std_n_trotters = np.std(n_trotters)
    max_n_trotters = max(n_trotters)
    min_n_trotters = min(n_trotters)

    exact_gs_energy = df['exact_gs_energy'].values[0]  
    bf_time = df['bf_time'].values[0]
    
    results[key] = {
        'n_spins': n_spins,
        'n_init': n_init,
        'n_iters': n_iters,
        'n_shots': n_shots,
        'delta_t': delta_t,
        'avg_min_energy': avg_min_energy,
        'std_min_energy': std_min_energy,
        'avg_BO_time': avg_BO_time,
        'std_BO_time': std_BO_time,
        'bf_time': bf_time,
        'beta': beta,
        'avg_beta': avg_beta,
        'std_beta': std_beta,
        'max_beta': max_beta,
        'min_beta': min_beta,
        'annealing_time': annealing_time,
        'avg_annealing_time': avg_annealing_time,
        'std_annealing_time': std_annealing_time,
        'max_annealing_time': max_annealing_time,
        'min_annealing_time': min_annealing_time,
        'gamma_0': gamma_0,
        'avg_gamma_0': avg_gamma_0,
        'std_gamma_0': std_gamma_0,
        'max_gamma_0': max_gamma_0,
        'min_gamma_0': min_gamma_0,
        'n_trotters': n_trotters,
        'avg_n_trotters': avg_n_trotters,
        'std_n_trotters': std_n_trotters,
        'max_n_trotters': max_n_trotters,
        'min_n_trotters': min_n_trotters,
        'exact_gs_energy': exact_gs_energy
    }


# Fix  n_init and n_iters and vary the number of spins
x = {}
y = {}
yerr = {}
avg_beta_d = {}
std_beta_d = {}
avg_annealing_time_d = {}
std_annealing_time_d = {}
avg_gamma_0_d = {}
std_gamma_0_d = {}
avg_n_trotters_d = {}
std_n_trotters_d = {}

new_keys = [f"{int(key.split('_')[1])}_{int(key.split('_')[2])}" for key in tables.keys()]
new_keys = list(set(new_keys))
data_energy = {key: [] for key in new_keys}
data_beta = {key: [] for key in new_keys}
data_annealing_time = {key: [] for key in new_keys}
data_gamma_0 = {key: [] for key in new_keys}
data_n_trotters = {key: [] for key in new_keys}

for key, result in results.items():
    for new_key in new_keys:
        n_spins = int(key.split('_')[0])
        n_init = int(key.split('_')[1])
        n_iters = int(key.split('_')[2])

        if f"{n_init}_{n_iters}" == new_key:
            data_energy[new_key].append((n_spins, abs(result['avg_min_energy'] - result['exact_gs_energy']), result['std_min_energy']))
            x[new_key], y[new_key], yerr[new_key] = zip(*data_energy[new_key])

            data_beta[new_key].append((n_spins, result['avg_beta'], result['std_beta']))
            _, avg_beta_d[new_key], std_beta_d[new_key] = zip(*data_beta[new_key])

            data_annealing_time[new_key].append((n_spins, result['avg_annealing_time'], result['std_annealing_time']))
            _, avg_annealing_time_d[new_key], std_annealing_time_d[new_key] = zip(*data_annealing_time[new_key])

            data_gamma_0[new_key].append((n_spins, result['avg_gamma_0'], result['std_gamma_0']))
            _, avg_gamma_0_d[new_key], std_gamma_0_d[new_key] = zip(*data_gamma_0[new_key])

            data_n_trotters[new_key].append((n_spins, result['avg_n_trotters'], result['std_n_trotters']))
            _, avg_n_trotters_d[new_key], std_n_trotters_d[new_key] = zip(*data_n_trotters[new_key])
            
            break


###################################################################
################## Generation of plots ############################
###################################################################

# Spins vs |E(SQA) - E(exact)|

plt.figure()
plt.title('Performance of BO for SQA')
plt.xlabel('Number of Spins')
plt.ylabel('|Estimated GS Energy - Brute Force energy|')

colors = ['blue', 'red', 'black', 'green', 'orange']

for i, key in enumerate(x):
    plt.errorbar(x[key], y[key], yerr[key], fmt='o', color=colors[i], capsize=5, label=f"BO initial samples: {key.split('_')[0]}, BO number of iterations: {key.split('_')[1]}")
    
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/figures/energies_plot_SQA_vs_BF.png')


# Spins vs hyperparameters

# 1. beta

plt.figure()
plt.title('Optimal Beta vs Number of Spins in BO-SQA')
plt.xlabel('Number of Spins')
plt.ylabel('Optimal Beta')

colors = ['blue', 'red', 'black', 'green', 'orange']

for i, key in enumerate(x):
    plt.errorbar(x[key], avg_beta_d[key], std_beta_d[key], fmt='o', color=colors[i], capsize=5, label=f"BO initial samples: {key.split('_')[0]}, BO number of iterations: {key.split('_')[1]}")
    
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/figures/opt_beta_BO_SQA.png')


# 2. annealing_time

plt.figure()
plt.title('Optimal annealing_time vs Number of Spins in BO-SQA')
plt.xlabel('Number of Spins')
plt.ylabel('Optimal annealing_time')

colors = ['blue', 'red', 'black', 'green', 'orange']

for i, key in enumerate(x):
    plt.errorbar(x[key], avg_annealing_time_d[key], std_annealing_time_d[key], fmt='o', color=colors[i], capsize=5, label=f"BO initial samples: {key.split('_')[0]}, BO number of iterations: {key.split('_')[1]}")
    
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/figures/opt_annealing_time_BO_SQA.png')


# 3. gamma_0

plt.figure()
plt.title('Optimal gamma_0 vs Number of Spins in BO-SQA')
plt.xlabel('Number of Spins')
plt.ylabel('Optimal gamma_0')

colors = ['blue', 'red', 'black', 'green', 'orange']

for i, key in enumerate(x):
    plt.errorbar(x[key], avg_gamma_0_d[key], std_gamma_0_d[key], fmt='o', color=colors[i], capsize=5, label=f"BO initial samples: {key.split('_')[0]}, BO number of iterations: {key.split('_')[1]}")    
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/figures/opt_gamma_0_BO_SQA.png')


# 4. n_trotters

plt.figure()
plt.title('Optimal n_trotters vs Number of Spins in BO-SQA')
plt.xlabel('Number of Spins')
plt.ylabel('Optimal n_trotters')

colors = ['blue', 'red', 'black', 'green', 'orange']

for i, key in enumerate(x):
    plt.errorbar(x[key], avg_n_trotters_d[key], std_n_trotters_d[key], fmt='o', color=colors[i], capsize=5, label=f"n_init: {key.split('_')[0]}, n_iters: {key.split('_')[1]}")
    
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/SQA/SQA_Tunning/Codes/Laboratory/SQA_tunning_temp_gamma_0_v1/ML_hyp_tuning/figures/opt_n_trotters_BO_SQA.png')