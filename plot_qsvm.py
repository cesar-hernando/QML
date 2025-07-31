'''
In this file, we generate plots for the Quantum Support Vector Machine (QSVM) analysis results.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the CSV file
device = 'myQLM'  # 'Qaptiva' or 'myQLM'
dataset = 'breast_cancer' # 'breast cancer' or 'wine'
file_name = f'2025-07-20_035634_PCA_encoding_anglex_evaluation_breast_cancer_myQLM.csv' # Change this to the appropriate file name  
path = f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/myQLM/myQLM/tutorials/quantum_machine_learning/tables_qsvm/{device}/{file_name}'
df = pd.read_csv(path)

components = df['Qubits']
encodings = df['Encoding']
accuracies = df['Accuracy']
fit_times = df['Fit_time']
predict_times = df['Pred_time']

# Create a figure of accuracies vs number of qubits for each encoding
plt.figure()
for i, encoding in enumerate(encodings.unique()):
    mask = df['Encoding'] == encoding
    plt.plot(df[mask]['Qubits'], df[mask]['Accuracy'], marker = 'o', label=encoding)
    
plt.title(f'QSVM Accuracy for {dataset} dataset')
plt.xlabel('Number of Reduced features (Qubits)')
plt.ylabel('Accuracy')
plt.xticks(components.unique())
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/myQLM/myQLM/tutorials/quantum_machine_learning/figures/qsvm/accuracy_plot_{file_name}.png')

# Create a figure of fit and predict times vs number of qubits for each encoding
plt.figure()
for i, encoding in enumerate(encodings.unique()):
    mask = df['Encoding'] == encoding
    plt.plot(df[mask]['Qubits'], df[mask]['Fit_time'], marker='o', label=f'{encoding} Fit Time')
    plt.plot(df[mask]['Qubits'], df[mask]['Pred_time'], marker='x', label=f'{encoding} Predict Time')
plt.title(f'QSVM Fit and Predict Times for {dataset} dataset')
plt.xlabel('Number of Reduced features (Qubits)')
plt.ylabel('Time (seconds)')
plt.xticks(components.unique())
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/myQLM/myQLM/tutorials/quantum_machine_learning/figures/qsvm/times_plot_{file_name}.png')