'''

Quantum Machine Learning (QML) Library in myQLM SDK:
----------------------------------------------------

In this file, we define different Quantum Machine Learning (QML) algorithms implemented in the SDK myQLM by Atos:

1. Quantum Support Vector Machine (QSVM) defines a quantum kernel used as a subroutine in a SVM, 
which is a supervised machine learning algorithm used for binary classification tasks. 

2. Quantum Circuit Born Machine (QCBM) is an unsupervised implicit generative machine learning model,
which leverages Born's postulate to express a probability distribution as a quantum circuit.

3. Quantum-Classical Convolutional Neural Network (QCCNN) is a supervised hybrid model that combines 
quantum convolutional layers with classical convolutional neural networks.

'''

import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from time import time
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from qat.lang.AQASM import Program, H, RX, RY, RZ, CNOT
from qat.qpus import get_default_qpu, PyLinalg
from qat.core import Observable, Term
from qat.plugins import ObservableSplitter
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
    


####################################################################################
############### Quantum-Classical Convolutional Neural Network #####################
####################################################################################

class QCCNN(ABC):
    def __init__(self, n_qubits, device, n_blocks, n_shots, optimizer_name, loss, learning_rate = 0.01, opt_model_path=None, np_arrays_path=None):

        if device.lower() == 'qaptiva':
            from qlmaas.qpus import LinAlg

        self.n_qubits = n_qubits
        self.device = device
        self.n_blocks = n_blocks
        self.n_shots = n_shots
        self.optimizer_name = optimizer_name
        self.loss = loss
        self.learning_rate = learning_rate
        self.opt_model_path = opt_model_path
        self.np_arrays_path = np_arrays_path


    @abstractmethod
    def quantum_conv_kernel_circuit(self, x):
        '''
        This method implements a quantum convolutional kernel circuit for a QCCNN.
        It applies a series of rotation gates and CNOT gates to the qubits based on the input data x and parameters params.
        The circuit is designed to encode the input data into the quantum state of the qubits.

        Parameters:
        -----------
        x (np.ndarray): 
            Input data vector of length n_qubits that is used as input angles of the rotation gates applied to each qubit in the feature map.


        Returns:
        --------
        expected_z_values (np.ndarray):
            The expectation values of the Z observable for each qubit after applying the quantum circuit.
        '''
        pass


    def quantum_conv_layer(self, image):
        '''
        Slides the quantum convolutional kernel over the image and applies the quantum circuit to each 2x2 region, and generates 4 channels of output.
        '''
        length_image = image.shape[0]  # Assuming the image is square
        output_image = np.zeros((length_image//2, length_image//2, 4))

        # Iterate over the pixels of the original image in steps of 2
        for j in range(0, length_image, 2):
            for k in range(0, length_image, 2):
                # Evaluates the quantum circuit on the 2x2 region of the image
                q_results = self.quantum_conv_kernel_circuit([image[j, k, 0], image[j, k + 1, 0], image[j + 1, k, 0], image[j + 1, k + 1, 0]])
                
                # Each qubit expectation value corresponds to a channel in the output image
                for c in range(4):
                    output_image[j // 2, k // 2, c] = q_results[c]

        return output_image


    def quantum_conv_preprocessing(self, train_images, test_images=None, save=True, params=None):
        '''
        Applies the quantum convolutional layer to each image in the dataset.
        '''
        if params is None:
            n_params = 3*self.n_qubits*self.n_blocks # Number of parameters in the quantum convolutional kernel
            self.params = np.random.rand(n_params) * np.pi  # Random parameters for the quantum circuit
        else:
            self.params = params

        if save:
            np.save(self.np_arrays_path + "quantum_random_params.npy", self.params)

        quantum_train_images = np.asarray([self.quantum_conv_layer(image=img) for img in train_images])
        if save:
            np.save(self.np_arrays_path + "quantum_train_images.npy", quantum_train_images)

        # If no test images are provided, return only the processed training images
        if test_images is None:
            return quantum_train_images, self.params
        
        quantum_test_images = np.asarray([self.quantum_conv_layer(image=img) for img in test_images])
        if save:
            np.save(self.np_arrays_path + "quantum_test_images.npy", quantum_test_images)

        return quantum_train_images, quantum_test_images, self.params
    
    
    @abstractmethod
    def classical_model(self):
        pass


    def train(self, preprocessing, train_images, train_labels, validation_images, validation_labels, batch_size, n_epochs, early_stop_crit=False, patience=None, min_delta=None):
        '''
        This method trains the QCCNN model using the quantum convolutional preprocessing and a classical model.
        '''

        if preprocessing:
            quantum_train_images, quantum_validation_images, _ = self.quantum_conv_preprocessing(train_images, validation_images)
        else:
            quantum_train_images = train_images
            quantum_validation_images = validation_images
        

        class_model = self.classical_model()

        if early_stop_crit:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta = min_delta, restore_best_weights = True)

            self.history = class_model.fit(
                quantum_train_images,
                train_labels,
                validation_data=(quantum_validation_images, validation_labels),
                batch_size=batch_size,
                epochs=n_epochs,
                verbose=2,
                callbacks=[early_stop]
            )
        else:
            self.history = class_model.fit(
                quantum_train_images,
                train_labels,
                validation_data=(quantum_validation_images, validation_labels),
                batch_size=batch_size,
                epochs=n_epochs,
                verbose=2
            )

        # Save the model if a path is specified
        if self.opt_model_path is not None:
            # Save the model to the specified path
            class_model.save(self.opt_model_path + 'qccnn_model_v0.keras')

        return self.history


    def plot_quantum_images(self, train_images, quantum_train_images, n_samples):
        '''
        Plot n_samples examples of the output images resulting from the quantum convolutional layer preprocessing
        '''
        n_samples = 4
        n_channels = self.n_qubits
        fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
        for k in range(n_samples):
            axes[0, 0].set_ylabel("Input")
            if k != 0:
                axes[0, k].yaxis.set_visible(False)
            axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

            # Plot all output channels
            for c in range(n_channels):
                axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
                if k != 0:
                    axes[c, k].yaxis.set_visible(False)
                axes[c + 1, k].imshow(quantum_train_images[k, :, :, c], cmap="gray")

        plt.tight_layout()
        plt.show = ()


    def plot_loss(self, c_history=None, fig_path=None, fig_name='qccnn_loss.png'):
        '''
        Plots two figures: one for the evolution of the training and validation accuracy of the model with and without the quantum convolutional layer, and the same figure but for the loss
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

        ax1.plot(self.history.history["accuracy"], "--r", label="Training accuracy with quantum layer")
        ax1.plot(self.history.history["val_accuracy"], "-b", label="Validation accuracy with quantum layer")
        if c_history is not None: 
            ax1.plot(c_history.history["accuracy"], "--c", label="Training accuracy without quantum layer")
            ax1.plot(c_history.history["val_accuracy"], "-g", label="Validation accuracy without quantum layer")
            
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(self.history.history["loss"], "--r", label="Training loss with quantum layer")
        ax2.plot(self.history.history["val_loss"], "-b", label="Validation loss with quantum layer")
        if c_history is not None: 
            ax2.plot(c_history.history["loss"], "--c", label="Training loss without quantum layer")
            ax2.plot(c_history.history["val_loss"], "-g", label="Validation loss without quantum layer")

        ax2.set_ylabel("Loss")
        ax2.set_ylim(top=2.5)
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.tight_layout()
        plt.show = ()

        if fig_path is not None:
            fig.savefig(fig_path + fig_name, dpi=300, bbox_inches='tight')
            print(f'Figure saved at {fig_path}' + fig_name)

    
    def optimize_quantum_params(self, train_images, train_labels, method='cobyla', max_iter=10, n_init=5, n_iters=20):
        '''
        Fixing the optimal weights and biases of the classical CNN, we vary the angles of the rotational gates in the quantum layer
        to minimize the training loss, using different training data. We implement wo different gradient-free optimization methods: 
        Cobyla and Bayesian Optimization. In the future, parameter shift rule or autodifferentiation could be implemented.
        '''
        params_0 = self.params
        n_params = len(params_0)

        lower_bound = 0.0
        upper_bound = np.pi

        cnn_model = tf.keras.models.load_model(self.opt_model_path + 'qccnn_model_v0.keras')

        train_accuracies = []
        train_losses = []

        def quantum_loss_function(params):
            self.params = params
            quantum_train_images, _ = self.quantum_conv_preprocessing(train_images, save=False, params=params)
            train_loss, train_accuracy = cnn_model.evaluate(quantum_train_images, train_labels, verbose=0)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            return train_loss
        

        if method.lower() == 'cobyla':
            # COBYLA needs constraints in the form g(x) >= 0
            constraints = []
            for i in range(n_params):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - lower_bound})  # x[i] >= lower
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: upper_bound - x[i]})  # upper - x[i] >= 0

            result = minimize(
                quantum_loss_function,
                params_0,
                method='COBYLA',
                constraints=constraints,
                options={'maxiter': max_iter, 'disp': True}
            )

            self.params = result.x
            np.save(self.np_arrays_path + "quantum_opt_params.npy", self.params)

            return result, train_accuracies, train_losses
        
        elif method.lower() == 'bayesian_optimization':
            min_vals = [0]*n_params
            max_vals = [np.pi]*n_params
            bounds = [(min_vals[i], max_vals[i]) for i in range(n_params)]

            bo = BayesianOptimizer(func=quantum_loss_function, bounds=bounds, n_init=n_init)
            X_opt, Y_opt, history = bo.optimize(n_iters=n_iters)
            best_idx = np.argmin(Y_opt)

            self.params = X_opt[best_idx]
            np.save(self.np_arrays_path + "quantum_opt_params.npy", self.params)

            return history, train_accuracies, train_losses

    
    def predict(self, preprocessing, test_images, test_labels):
        '''
        Evaluates the performance of the quantum enhanced convolutional network on the test set.
        '''
        if preprocessing:
            quantum_test_images, _ = self.quantum_conv_preprocessing(test_images, save=False, params=self.params)
        else:
            quantum_test_images = test_images

        cnn_model = tf.keras.models.load_model(self.opt_model_path + 'qccnn_model_v0.keras')
        test_loss, test_accuracy = cnn_model.evaluate(quantum_test_images, test_labels, verbose=0)
        return test_loss, test_accuracy
    

    @staticmethod
    def plot_q_train_loss(train_accuracies, train_losses, fig_path=None, fig_name='qccnn_quantum_loss.png'):
        '''
        Plots the training accuracy during the optimization of the angles in the quantum convolutional layer.
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

        ax1.plot(train_accuracies, "-or")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Epoch")
        ax1.set_title("Evolution of training accuracy in quantum ansatz training")

        ax2.plot(train_losses, "-or")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(top=2.5)
        ax2.set_xlabel("Epoch")
        ax1.set_title("Evolution of training loss in quantum ansatz training")
    
        plt.tight_layout()
        plt.show = ()

        if fig_path is not None:
            fig.savefig(fig_path + fig_name, dpi=300, bbox_inches='tight')
            print(f'Figure saved at {fig_path}' + fig_name)
    



class Default_QCCNN(QCCNN):
    def quantum_conv_kernel_circuit(self, x):
        '''
        This method implements a quantum convolutional kernel circuit for a custom QCCNN. It applies first an angle encoding 
        feature map to map the input data (portion of the image) to a quantum tate, and then applies series of rotation gates 
        with random angles and CNOT gates to the qubits.

        Parameters:
        -----------
        x (np.ndarray): 
            Input data vector of length n_qubits that is used as input angles of the rotation gates applied to each qubit in the feature map.


        Returns:
        --------
        expected_z_values (np.ndarray):
            The expectation values of the Z observable for each qubit after applying the quantum circuit.
        '''

        angle_encoding = Angle_Encoding(self.n_qubits)
        qprogram, qubits = angle_encoding.get_quantum_program(x)
        
        for b in range(self.n_blocks):
            for i, qubit in enumerate(qubits):
                qprogram.apply(RZ(self.params[3 * (i + b * self.n_qubits)]), qubit)
                qprogram.apply(RX(self.params[3 * (i + b * self.n_qubits) + 1]), qubit)
                qprogram.apply(RZ(self.params[3 * (i + b * self.n_qubits) + 2]), qubit)
            for i, qubit in enumerate(qubits):
                if (i + 1 + b) % self.n_qubits != i:
                    qprogram.apply(CNOT, qubit, qubits[(i + b + 1) % self.n_qubits])
                else:
                    qprogram.apply(CNOT, qubit, qubits[(i + 1) % self.n_qubits])
        
        circuit = qprogram.to_circ()

        expected_z_values = np.zeros(self.n_qubits)

        for i in range(self.n_qubits):
            obs = Observable(self.n_qubits)
            obs.add_term(Term(1, "Z", [i]))

            # Submit the circuit to the QPU, measuring the expectation value of the observable
            if self.device.lower() == 'qaptiva':
                qpu = ObservableSplitter() | LinAlg()
            elif self.device.lower() == 'myqlm':
                qpu = ObservableSplitter() | PyLinalg()
            else:
                raise ValueError(f'Unknown device: {self.device}')

            # Create a quantum job and use a finite number if it is specified in the class instance
            if self.n_shots is None:
                job = circuit.to_job("OBS", observable=obs)
            else:
                job = circuit.to_job("OBS", observable=obs, nbshots=self.n_shots)

            # Submit the job and obtain the probability of measuing all qubits in state 0
            expected_z_values[i] = qpu.submit(job).value

        return expected_z_values
    

    def classical_model(self):
        '''
        Initializes and returns a custom Keras model
        which is ready to be trained
        '''
        
        model = tf.keras.models.Sequential([

            # First convolutional layer
            tf.keras.layers.Conv2D(filters=50, kernel_size=(2, 2), activation='relu'),
                    
            # First pooling layer
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second convolutional layer
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
            
            # Second pooling layer
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten to feed into dense layer
            tf.keras.layers.Flatten(),
            
            # Fully connected dense layer
            tf.keras.layers.Dense(64, activation='relu'),
            
            # Final output layer with 10 units for classification
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        if self.optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            pass # Add other optimizers as needed

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=["accuracy"],
        )

        return model
    


####################################################################################
################################ Feature Maps ######################################
####################################################################################

class Feature_Map(ABC):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    @abstractmethod
    def get_quantum_program(self, a, qprogram=None, qubits=None):
        pass

class Angle_Encoding(Feature_Map):
    def __init__(self, n_qubits, axis='x'):
        super().__init__(n_qubits)
        self.axis = axis

    def get_quantum_program(self, a, qprogram=None, qubits=None):

        if qprogram is None:
        # Create a quantum program with the specified number of qubits if qprogram is not provided
            qprogram = Program()
            qubits = qprogram.qalloc(self.n_qubits)
        
        for i, qubit in enumerate(qubits):
            if self.axis.lower() == 'x':
                qprogram.apply(RX(a[i]), qubit)
            elif self.axis.lower() == 'y':
                qprogram.apply(RY(a[i]), qubit)
            elif self.axis.lower() == 'z':
                qprogram.apply(H, qubit)
                qprogram.apply(RZ(a[i]), qubit)
            else:
                raise ValueError(f'Invalid axis {self.axis}. Select x, y or z')
            
        return qprogram, qubits
    
class ZZ_Feature_Map(Feature_Map):
    def get_quantum_program(self, a, qprogram=None, qubits=None):

        if qprogram is None:
            # Create a quantum program with the specified number of qubits if qprogram is not provided
            qprogram = Program()
            qubits = qprogram.qalloc(self.n_qubits)
        
        for i, qubit in enumerate(qubits):
            qprogram.apply(H, qubit)
            qprogram.apply(RZ(2*a[i]), qubit)

        for i, control in enumerate(qubits):
            for j, target in enumerate(qubits[i+1:], start=i+1):
                qprogram.apply(CNOT, control, target)
                qprogram.apply(RZ(2*(np.pi-a[i])*(np.pi-a[j])), target)
                qprogram.apply(CNOT, control, target)

        return qprogram, qubits
    

    ############################################################
    ######### Classical ML algorithms ##########################
    ############################################################

class BayesianOptimizer:
    def __init__(self, func, bounds, n_init=5, kernel=None, xi=0.01, seed=42):
        """
        Parameters:
            sqa : class
                Contains the black-box function to optimize.
            bounds : list of tuples
                Bounds for each dimension [(min, max), ...].
            n_init : int
                Number of initial random evaluations.
            kernel : sklearn kernel or None
                Kernel for the Gaussian Process.
            xi : float
                Exploration-exploitation parameter for EI.
        """

        np.random.seed(seed)

        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.xi = xi
        self.kernel = kernel if kernel else Matern(nu=2.5)
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-3, normalize_y=True)

        # Initialize with random samples
        self.X_sample = self.random_sample(n_init)
        self.Y_sample = self.evaluate(self.X_sample)

    def random_sample(self, n):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n, self.dim))

    def evaluate(self, X):
        return np.array([self.func(x) for x in X]).reshape(-1, 1)

    def expected_improvement(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu_sample_opt = np.min(self.Y_sample)
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    def suggest(self, n_candidates=1000, n_restarts=10, mode=1):

        def min_obj(x):
                x = np.atleast_2d(x)
                return -self.expected_improvement(x)

        if mode == 0: # random search
            X_grid = self.random_sample(n_candidates)
            ei = self.expected_improvement(X_grid)
            return X_grid[np.argmax(ei)]
        else:
            best_x = None
            best_ei = -np.inf

            for _ in range(n_restarts):
                x0 = self.random_sample(1).flatten()  # random initial point
                res = minimize(min_obj, x0=x0, bounds=self.bounds, method="L-BFGS-B")

                if res.success:
                    ei_val = -res.fun  # remember we minimized the negative
                    if ei_val > best_ei:
                        best_ei = ei_val
                        best_x = res.x

            return best_x
        

    def step(self):
        self.model.fit(self.X_sample, self.Y_sample)
        x_next = self.suggest().reshape(1, -1)
        y_next = self.evaluate(x_next)
        self.X_sample = np.vstack((self.X_sample, x_next))
        self.Y_sample = np.vstack((self.Y_sample, y_next))
        return x_next.squeeze(), y_next.item()

    def optimize(self, n_iters=20, verbose=True):
        history = []
        for i in range(n_iters):
            x, y = self.step()
            if verbose:
                print(f"Iter {i+1}: x = {x}, f(x) = {y:.4f}")
            history.append((x, y))
        return self.X_sample, self.Y_sample, history