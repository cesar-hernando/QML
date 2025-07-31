'''
In this file, we analyse the Quantum Support Vector Machine (QSVM) applied to two datasets: a wine dataset and a breast cancer dataset.
We will analyses the effct of performing PCA on the datasets to reduce the number of attributes, as well as th effect of the quantum kernel used.
'''

# Import the necessary libraries and the self-made QSVM class
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import load_wine
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from itertools import combinations
from tqdm.notebook import tqdm

from myqml import QSVM

# Select the device used and connect to the server if 'qaptiva' is selected
device= 'Qaptiva'  # 'Qaptiva' or 'myQLM'

if device.lower() == 'qaptiva':
    from qat.qlmaas import QLMaaSConnection
    conn = QLMaaSConnection(hostname="qlm35e.neasqc.eu", check_host=False)

# Define parameters
dataset = 'wine'  # 'wine' or 'breast_cancer'
train_size = 0.8  # Proportion of the dataset to include in the train split

# Load the dataset
if dataset.lower() == 'wine':
    x, y = load_wine(return_X_y=True)
    # Filter the dataset to keep only two types of wines
    x = x[:59+71,:]
    y = y[:59+71]
    
elif dataset.lower() == 'breast_cancer':
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    x = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 

    # Convert to numpy arrays 
    x = x.to_numpy()
    y = y.to_numpy()
    y = y.ravel()
    

# Get the number of examples and features
n_examples = x.shape[0]
n_features = x.shape[1]

# Split the data into training and test sets, and scale the features using only the training data to prevent data leakage
seed = 1
np.random.seed(seed)
x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size = train_size, random_state=seed)
scaler = MaxAbsScaler()
x_tr = scaler.fit_transform(x_tr)
x_test = scaler.transform(x_test)
x_test = np.clip(x_test, 0, 1)

# Vary the number of components for PCA and the encoding type
if dataset.lower() == 'wine':
    n_components_list = [i for i in range(1, n_features+1)] # Include n_components = n_features but it will be treated as a special case
elif dataset.lower() == 'breast_cancer':
    n_components_list = [2, 4, 6, 8, 10, 12] 

encoding_list = ['x', 'y', 'z','zz']  # Firts three are angle encoding, last one is ZZ feature map

# Prepare to store results
encodings = []
features = []
fit_times = []
predict_times = []
accuracies = []
TN_list = []
FP_list = []
FN_list = []
TP_list = []
dates = []
times = []
PCA_list = []
devices = [device] * len(n_components_list) * len(encoding_list)
n_examples_list = [n_examples] * len(n_components_list) * len(encoding_list)
train_size_list = [train_size] * len(n_components_list) * len(encoding_list)

print(f"Starting evaluation for dataset: {dataset}, device: {device}...")

for n_components in n_components_list:
    for encoding in encoding_list:
        # Apply PCA to reduce the number of attributes
        if n_components < n_features:
            pca = PCA(n_components=n_components)
            x_tr_pca = pca.fit_transform(x_tr)
            x_test_pca = pca.transform(x_test)
        else:
            x_tr_pca = x_tr
            x_test_pca = x_test

        # Define the QSVM algorithm with the specified parameters
        qsvm = QSVM(n_qubits=n_components,  
                    device=device,
                    kernel_circuit_label='angle_encoding' if encoding != 'zz' else 'zz_encoding',
                    angle_encoding_type=encoding if encoding != 'zz' else None)

        # Train the QSVM model
        start_fit = time.time()
        qsvm.fit(x_tr_pca, y_tr)
        end_fit = time.time()
        fit_time = end_fit - start_fit
        fit_times.append(fit_time)

        # Predict on the test set
        start_predict = time.time()
        y_pred = qsvm.predict(x_test_pca)
        end_predict = time.time()
        predict_time = end_predict - start_predict
        predict_times.append(predict_time)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Calculate confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        TN = conf_mat[0, 0]  # True Negatives
        FP = conf_mat[0, 1]  # False Positives
        FN = conf_mat[1, 0]  # False Negatives
        TP = conf_mat[1, 1]  # True Positives
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        TP_list.append(TP)

        # Store the encoding, number of features, date, and time
        if encoding == 'zz':
            encodings.append('zz_feature_map')
        else:
            encodings.append(f'angle_encoding_{encoding}')

        features.append(n_components)
        PCA_list.append(True if n_components < n_features else False)
        dates.append(str(pd.Timestamp.now().date()))
        times.append(str(pd.Timestamp.now().time().replace(microsecond=0)))
        print(f"Execution for n_components={n_components}, encoding={encoding} completed with accuracy: {accuracy:.4f}, fit time: {fit_time:.4f}s, predict time: {predict_time:.4f}s")
        print(f"Confusion Matrix: {conf_mat}\n")


# Create a DataFrame to store the results
df = pd.DataFrame(list(zip(dates, times, features, devices, features, n_examples_list, train_size_list, encodings, PCA_list, accuracies, TN_list, FP_list, FN_list, TP_list, fit_times, predict_times)), 
                  columns=['Date', 'Time', 'Qubits', 'Emulator', 'Features', 'Samples', 'Train_prop', 'Encoding', 'PCA', 'Accuracy', 'TN', 'FP', 'FN', 'TP', 'Fit_time', 'Predict_time'])

# Save the DataFrame to a CSV file
date_str = pd.Timestamp.now().date().isoformat()
time_str = pd.Timestamp.now().time().replace(microsecond=0).isoformat().replace(':', '')
file_name = f'{date_str}_{time_str}_PCA_encoding_anglex_evaluation_{dataset}_{device}.csv'
path = f'C:/Users/a943763/OneDrive - ATOS/Documentos/Python/myQLM/myQLM/tutorials/quantum_machine_learning/tables_qsvm/{device}/{file_name}'
df.to_csv(path)

