# Quantum Machine Learning (QML) in myQLM (Eviden, Atos Group)

This repository is the result of a summer internship at Eviden in 2025 and contains the implementation of three QML methods: Quantum Support Vector Machine (QSVM), Quantum Circuit Born Machine (QCBM) and Quantum-Classical Convolutional Neural Network (QCCNN). The code of these methods is in the myqlm.py file. Additionally, it includes an implementation of Bayesian Optimization (BO) applied to hyperparameter tuning for the quantum-inspired Simulated Quantum Annealing (SQA) algorithm, designed to solve Binary Combinatorial Optimization problems.

- QSVM: A quantum-enhanced supervised learning method for binary classification problems, which implements a quantum circuit kernel, as part of the SVM routine. For more information of the algorithm, read through the example_qsvm.ipynb tutorial.

- QCBM: A quantum-enhanced unsupervised generative learning method that learns the underlying probability distribution of a dataset and generate samples that resemble the training data. It leverages the Born's posutlate to encode the probability distribution in a parametrized quantum circuit. For more information of the algorithm, read through the qcbm_bars_stripes.ipynb tutorial.

- QCCNN: A quantum-enhanced supervised learning method for image classification into multiple classes. It combines a quantum convolutional layer with a CNN implemented in tensorflow. For more information of the algorithm, read through the qccnn_mnist.ipynb tutorial.

- BO applied to hyperparameter tuning of SQA: Information in the BO-SQA folder's README.
