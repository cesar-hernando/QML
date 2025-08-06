'''
In this file, we define different Quantum Machine Learning (QML) algorithms implemented in the SDK myQLM by Atos:

1. Quantum Support Vector Machine (QSVM) defines a quantum kernel used as a subroutine in a SVM, 
which is a supervised machine learning algorithm used for binary classification tasks. 

2. Quantum Circuit Born Machine (QCBM) is an unsupervised implicit generative machine learning model,
which leverages Born's postulate to express a probability distribution as a quantum circuit.

'''

import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
from scipy.optimize import minimize
from time import time
from sklearn.svm import SVC

from qat.lang.AQASM import Program, H, RX, RY, RZ, CNOT
from qat.qpus import get_default_qpu
#from qlmaas.qpus import LinAlg # Comment when using myQLM


#############################################################
############## Quantum Suport Vector Machine ################
#############################################################

class QSVM(SVC):

    '''
    Class to solve a Supervised Machine Learning binary classification problem using the 
    Quantum Support Vector Machine (QSVM) algorithm. It defines the qkernel method which 
    is used as input of the SVC class (parent class) of scikit-learn that implements a 
    classical SVM algorithm. Thus, all the quantumness is embedded in the definition of 
    the kernel.

    Parameters:
    ----------
    n_qubits (int): 
        Number of qubits used for the quantum circuit, which must be equal to the number of features of the dataset

    device (str): 
        Indicates whether the quantum circuits are executed locally (myQLM) or sent to a Qaptiva emulator as batches.

    kernel_circuit_label (str): 
        Encoding type for the kernel circuit. Can be either 'angle_encoding', 'zz_encoding' or None if a custom_kernel is used.

    angle_encoding_type (str): 
        Axis of rotation employed for the rotation gates of the quantum circuit ('x', 'y' or 'z'), if 'angle_encoding' is utilized. It can be None is 'zz_encoding' is selected.
    
    n_shots (int):
        Number of executions of each quantum circuit to estimate the probability of measuring all qubits in state 0. If n_shots is not indicated, the probability is calculated
        exactly (with state vector simulation) without shot noise.   

    custom_kernel (function): 
        Given two input vectors, the function must create a kernel circuit and return an obbject of the class Program.

    '''

    def __init__(self, n_qubits, device='Qaptiva', kernel_circuit_label='angle_encoding', angle_encoding_type='x', n_shots = None, custom_kernel=None):
        
        self.n_qubits = n_qubits
        self.kernel_circuit_label = kernel_circuit_label
        self.device = device
        self.custom_kernel = custom_kernel

        if kernel_circuit_label == 'angle_encoding':
            self.angle_encoding_type = angle_encoding_type
        else:
            self.angle_encoding_type = None            

        if n_shots is not None:
            self.n_shots = n_shots
        else:
            self.n_shots = None

        super().__init__(kernel=self.qkernel)


    def kernel_circuit(self, a, b):
        '''
        Given two input vectors (samples of the dataset), this method creates a quantum circuit to implement the kernel specified 
        by the attributes **kernel_circuit_label** and **angle_encoding_type**  and returns the probability of measuring all qubits in state 0.

        Parameters:
        -----------
        a (np.ndarray): 
            Input vector of length n_qubits that is used as input angles of the rotation gates applied to each qubit in the feature map.
        b (np.ndarray): 
            Input vector of length n_qubits that is used as input angles of the rotation gates applied to each qubit in the inverse feature map.

        Returns: 
        --------
        probability_all_zeros (float):
            The probability of measuring all qubits in zero.

        '''
            
        # Create a quantum program with the specified number of qubits
        qprogram = Program()
        qubits = qprogram.qalloc(self.n_qubits)

        if self.kernel_circuit_label == 'angle_encoding':

            # Feature map for first input vector a
            for i, qubit in enumerate(qubits):
                if self.angle_encoding_type == 'x':
                    qprogram.apply(RX(a[i]), qubit)
                elif self.angle_encoding_type == 'y':
                    qprogram.apply(RY(a[i]), qubit)
                elif self.angle_encoding_type == 'z':
                    qprogram.apply(H, qubit)
                    qprogram.apply(RZ(a[i]), qubit)
                else:
                    raise ValueError(f'Unknown angle encoding type: {self.angle_encoding_type}')
                
            # Inverse feature map for second input vector b
            for i, qubit in enumerate(qubits):
                if self.angle_encoding_type.lower() == 'x':
                    qprogram.apply(RX(-b[i]), qubit)
                elif self.angle_encoding_type.lower() == 'y':
                    qprogram.apply(RY(-b[i]), qubit)
                elif self.angle_encoding_type.lower() == 'z':
                    qprogram.apply(RZ(-b[i]), qubit)
                    qprogram.apply(H, qubit)


        elif self.kernel_circuit_label == 'zz_encoding':

            # Feature map for first input vector a
            for i, qubit in enumerate(qubits):
                qprogram.apply(H, qubit)
                qprogram.apply(RZ(2*a[i]), qubit)

            for i, control in enumerate(qubits):
                for j, target in enumerate(qubits[i+1:], start=i+1):
                    qprogram.apply(CNOT, control, target)
                    qprogram.apply(RZ(2*(np.pi-a[i])*(np.pi-a[j])), target)
                    qprogram.apply(CNOT, control, target)

            # Inverse feature map for second input vector b
            for i in range(self.n_qubits-2, -1, -1):
                for j in range(self.n_qubits-1, i, -1):
                    qprogram.apply(CNOT, qubits[i], qubits[j])
                    qprogram.apply(RZ(-2*(np.pi-b[i])*(np.pi-b[j])), qubits[j])
                    qprogram.apply(CNOT, qubits[i], qubits[j])

            for i, qubit in enumerate(qubits):
                qprogram.apply(RZ(-2*b[i]), qubit)
                qprogram.apply(H, qubit)


        elif self.custom_kernel is not None:
            qprogram = self.custom_kernel(a,b)
                
        # Compile the full circuit
        circuit = qprogram.to_circ()

        # Submit the circuit to the QPU
        if self.device.lower() == 'qaptiva':
            qpu = LinAlg()
        elif self.device.lower() == 'myqlm':
            qpu = get_default_qpu()
        else:
            raise ValueError(f'Unknown device: {self.device}')

        # Create a quantum job and use a finite number if it is specified in the class instance
        if self.n_shots is None:
            job = circuit.to_job()
        else:
            job = circuit.to_job(nbshots=self.n_shots)

        # Submit the job and obtain the probability of measuing all qubits in state 0
        result = qpu.submit(job)
        probability_all_zeros = result[0].probability
            
        return probability_all_zeros
            
            
    def qkernel(self, A, B):
        '''
        This method iterates over all pairs of samples of the dataset and calls kernel_circuit to obtain the elements of the kernel matrix, which is returned.

        Parameters:
        -----------
        A: Iterable that contains the sample vectors used as inputs of the feature map.
        
        B: Iterable that contains the sample vectors used as inputs of the inverse feature map.

        Returns:
        --------
        kernel_matrix (np.ndarray):
            Kernel matrix of dimension n_samples x n_samples composed of the results of applying the quantum kernel circuit to every pair of samples of the dataset.

        '''

        # Evaluate the kernel matrix
        kernel_matrix = np.array([[self.kernel_circuit(a, b) for b in B] for a in A])     
        return kernel_matrix



#############################################################################
####################### Quantum Circuit Born Machine ########################
#############################################################################

class QCBM:
    def __init__(self, n_qubits, basis, n_blocks, n_shots, device, sigma_list_kernel, ansatz_mode=0, execution_mode=0, dimension=None):

        self.n_qubits = n_qubits
        self.basis = basis
        self.n_blocks = n_blocks
        self.n_shots = n_shots
        self.device = device
        self.ansatz_mode = ansatz_mode
        self.execution_mode = execution_mode
        self.sigma_list_kernel = sigma_list_kernel
        self.dimension = dimension

        if device.lower() == 'qaptiva':
            from qlmaas.qpus import LinAlg

        if ansatz_mode == 0:
            self.n_params = 3*n_qubits*n_blocks

        else: # A different ansatz could be implemented
            pass

    def ansatz(self, params):
        """Quantum Circuit Born Machine ansatz"""

        qprogram = Program()
        qubits = qprogram.qalloc(self.n_qubits)

        if self.ansatz_mode == 0:

            for b in range(self.n_blocks):
                for i, qubit in enumerate(qubits):
                    qprogram.apply(RZ(params[3 * (i + b * self.n_qubits)]), qubit)
                    qprogram.apply(RX(params[3 * (i + b * self.n_qubits) + 1]), qubit)
                    qprogram.apply(RZ(params[3 * (i + b * self.n_qubits) + 2]), qubit)
                for i, qubit in enumerate(qubits):
                    if (i + 1 + b) % self.n_qubits != i:
                        qprogram.apply(CNOT, qubit, qubits[(i + b + 1) % self.n_qubits])
                    else:
                        qprogram.apply(CNOT, qubit, qubits[(i + 1) % self.n_qubits]) 

        else:
            pass

        # Compile the full circuit
        circuit = qprogram.to_circ()

        # Submit the circuit to the QPU
        if self.device.lower() == 'qaptiva':
            qpu = LinAlg()
        elif self.device.lower() == 'myqlm':
            qpu = get_default_qpu()
        else:
            raise ValueError(f'Unknown device: {self.device}')

        # Execute quantum circuit in the selected mode
        if self.execution_mode == 1: # Realistic execution mode
            job = circuit.to_job(nbshots=self.n_shots)
            result = qpu.submit(job)
            samples = []
            for sample in result:
                count = round(sample.probability * self.n_shots)
                samples.extend([sample.state] * count)

            random.shuffle(samples)  # Shuffle samples to ensure randomness

        else: # Unrealistic but more efficient mode
            job = circuit.to_job()
            result = qpu.submit(job)
            states = []
            probs = []
            for sample in result:
                states.append(sample.state)
                probs.append(sample.probability)
            
            samples = np.random.choice(np.array(states), self.n_shots, p=np.array(probs))

        return samples, circuit

    def estimate_probs(self, params):
        '''
        This method estimates all probabilities from circuit samples.
        '''

        # Run the circuit and get bitstring samples
        samples, _ = self.ansatz(params)

        # Convert bitstrings to decimal indices
        samples_int = [int("".join(str(int(b)) for b in sample), 2) for sample in samples]

        # Compute histogram
        probs = np.zeros(2**self.n_qubits)
        for i in samples_int:
            probs[i] += 1

        return probs / self.n_shots
    

    # Multi-RBF kernel
    @staticmethod
    def multi_rbf_kernel(x, y, sigma_list):
        ndim = x.ndim
        if ndim == 1:
            exponent = np.abs(x[:, None] - y[None, :])**2
        elif ndim == 2:
            exponent = ((x[:, None, :] - y[None, :, :])**2).sum(axis=2)
        else:
            raise ValueError("Unsupported dimension")

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma)
            K += np.exp(-gamma * exponent)
        return K
    

    # Kernel expectation for MMD
    @staticmethod
    def kernel_expectation(px, py, kernel_matrix):
        return px @ kernel_matrix @ py
    

    def loss_function(self, params, target_probs, kernel_matrix):
        # Estimate probabilities from the circuit
        p_circuit = self.estimate_probs(params)
        # Compute kernel expectations
        K_p_target = self.kernel_expectation(p_circuit, target_probs, kernel_matrix)
        K_target_target = self.kernel_expectation(target_probs, target_probs, kernel_matrix)
        K_p_p = self.kernel_expectation(p_circuit, p_circuit, kernel_matrix)
        # Compute MMD
        mmd = K_p_p -2*K_p_target + K_target_target 
        return mmd
    

    def gradient(self, theta, kernel_matrix, target_probs):
        grad = []
        probs = self.estimate_probs(theta)

        for i in range(len(theta)):
            # Positive shift
            theta[i] += np.pi / 2
            probs_pos = self.estimate_probs(theta)

            # Negative shift
            theta[i] -= np.pi
            probs_neg = self.estimate_probs(theta)

            # Recover original
            theta[i] += np.pi / 2

            # Gradient component via finite differences
            grad_pos = self.kernel_expectation(probs, probs_pos, kernel_matrix) - self.kernel_expectation(probs, probs_neg, kernel_matrix)
            grad_neg = self.kernel_expectation(target_probs, probs_pos, kernel_matrix) - self.kernel_expectation(target_probs, probs_neg, kernel_matrix)
            grad.append(grad_pos - grad_neg)

        return np.array(grad)

    def fit(self, method="L-BFGS-B", learning_rate=0.1, tol=1e-5, max_iter=20, g_tol=1e-10, f_tol=0, x_tr=None, target_probs=None):

        # Initialize random theta
        params_0 = np.random.rand(self.n_params) * np.pi

        if target_probs is None:
            # Compute target probability distribution from the samples
            target_probs = np.zeros(2**self.n_qubits)
            for sample in x_tr:
                target_probs[int(sample, 2)] += 1
            target_probs = target_probs.astype(np.float64)
            target_probs /= len(x_tr)
        
        kernel_matrix = self.multi_rbf_kernel(self.basis, self.basis, self.sigma_list_kernel)
        # Define loss with fixed circuit and kernel
        loss_fn = partial(self.loss_function, target_probs=target_probs, kernel_matrix=kernel_matrix)
        grad_fn = partial(self.gradient, kernel_matrix=kernel_matrix, target_probs=target_probs)     

        # Track progress
        step = [0]
        tracking_cost = []

        def callback(x, *args, **kwargs):
            cost = loss_fn(x)
            tracking_cost.append(cost)
            step[0] += 1
            print(f"Step {step[0]}, Loss: {cost:.6f}")
        
        start_time = time()

        if method == "Adam":
            # Manual Adam implementation
            params = params_0.copy()
            m = np.zeros_like(params)
            v = np.zeros_like(params)
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            for t in range(1, max_iter + 1):
                g = grad_fn(params)
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                callback(params)

            result = {'x': params, 'fun': loss_fn(params), 'nit': t, 'success': True}
            self.opt_params = params
            
        elif method == "L-BFGS-B":
            result = minimize(
                loss_fn,
                params_0,
                method=method,
                jac=grad_fn,
                tol=tol,
                options={"maxiter": max_iter, "disp": 0, "gtol": g_tol, "ftol": f_tol},
                callback=callback
            )

            self.opt_params = result.x

        end_time = time()

        print(f"Optimization using {method} completed in {end_time - start_time:.2f} seconds.")

         

        return result, tracking_cost, end_time - start_time
    
    
    @staticmethod
    def plot_loss(tracking_cost):
        plt.figure()
        plt.plot(tracking_cost)
        plt.xlabel("Step")
        plt.ylabel("MMD Loss")
        plt.title("Evolution of MMD loss in QCBM")
        plt.grid(True)
        plt.show = ()


    def plot_generated_distribution(self, samples):
        # Convert bitstrings to decimal indices
        samples_int = [int("".join(str(int(b)) for b in sample), 2) for sample in samples]

        # Compute histogram
        probs = np.zeros(2**self.n_qubits)
        for i in samples_int:
            probs[i] += 1

        probs /= self.n_shots

        plt.plot(probs, 'r-')
        plt.title("Final QCBM Output Distribution")
        plt.xlabel("Bitstring index")
        plt.ylabel("Probability")
        plt.title("Probability distribution of generated data")
        plt.grid(True)
        plt.show = ()


    def generate_samples(self, n_samples):
        aux = self.n_shots
        self.n_shots = n_samples
        samples, _ = self.ansatz(self.opt_params)
        samples_matrix = np.array([np.array([int(bit) for bit in s], dtype=np.int8).reshape(*self.dimension) for s in samples])
        self.n_shots = aux
        return samples_matrix, samples


    def plot_generated_samples(self, samples_matrix):
        size = (int(np.ceil(np.sqrt(len(samples_matrix)))), int(np.ceil(np.sqrt(len(samples_matrix)))))
        plt.figure(facecolor='#777777')
        gs = plt.GridSpec(*size)

        for i in range(size[0]):
            for j in range(size[1]):
                if i*size[1]+j == len(samples_matrix): break
                plt.subplot(gs[i,j]).imshow(samples_matrix[i*size[1]+j], vmin=0, vmax=1)
                plt.axis('equal')
                plt.axis('off')
        plt.show = ()


    @staticmethod
    def calculate_metrics(samples_matrix, x_tr, validity_fn, n_sols, cost_fn):
        cost = cost_fn(samples_matrix)

        valid_indices = np.array([int(validity_fn(samples_matrix)[i]) for i in range(len(samples_matrix))])
        valid_samples = np.array([samples_matrix[i] for i,index in enumerate(valid_indices) if index==1])
        
        valid_unseen_samples = np.array([
            sample for sample in valid_samples
            if not any(np.array_equal(sample, x) for x in x_tr)
        ])
        
        unseen_samples = np.array([
            sample for sample in samples_matrix
            if not any(np.array_equal(sample, x) for x in x_tr)
        ])

        # Convert to tuple to use set for uniqueness
        unique_valid_unseen_samples = np.unique(valid_unseen_samples, axis=0)

        precision = len(valid_samples) / len(samples_matrix)
        fidelity = len(valid_unseen_samples) / len(unseen_samples) if len(unseen_samples) > 0 else 0
        rate = len(valid_unseen_samples) / len(samples_matrix)
        coverage = len(unique_valid_unseen_samples) / (n_sols + len(x_tr))

        metrics = {
            'precision': precision,
            'fidelity': fidelity,
            'rate': rate,
            'coverage': coverage,
            'average cost': cost
        }

        return metrics



