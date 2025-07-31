'''
In this file we implement the SQA algorithm
'''

# Import general use libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import networkx as nx
import pickle # Save and load graphs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# Import myQLM modules
from qat.opt import Ising
from qat.core.spins import integer_to_spins
from qat.core import Variable, Observable, Term
from qlmaas.qpus import SQAQPU




#############################################################
###################### SQA_solver ###########################
#############################################################

class SQA_solver:

    def __init__(self, n_spins, n_shots, delta_t):
        self.n_spins = n_spins
        self.n_shots = n_shots
        self.delta_t = delta_t

    def solve(self, hyperparameters):
        '''
        This function estimates the ground state energy of an Ising Hamiltonian using the
        Simulated Quantum Annealing algorithm.

        Parameters 
        -------
        hyperparameters: list
            List containing the hyperparameters for the SQA algorithm in the following order:
            [beta, annealing_time, gamma_0, n_trotters]
        
        beta: float
            Inverse temperature

        annealing_time: float
            Duration over which the quantum system evolves from initial to final Hamiltonian

        gamma_0: float
            Initial value of gamma schedule (used to simulate QA/tunneling)

        n_trotters: int
            Number of copies of each spins (used to simulate quantumness)

        Returns
        -------
        energy: float
            Estimate of the ground state energy

        '''

        # Unpack hyperparameters
        beta = hyperparameters[0]
        annealing_time = hyperparameters[1]
        gamma_0 = hyperparameters[2]
        n_trotters = hyperparameters[3]

        # Create an instance of the Ising class
        ising_problem = Ising(J=self.J, h=self.h, offset_i=0)

        # Initialitate variable of the functions
        t = Variable("t", float) 

        # Schedule representing a linear decrease from temp_max to temp_min (n_steps points)
        # In this case temp_max = temp_min = temp for a constant function
        temp_t  = (1/beta)*(t/annealing_time) + (1/beta)*(1 - t/annealing_time)  

        # Same schedule as temperature
        gamma_t = gamma_0 * (1 - t/annealing_time) 

        # Solve the problem with SQA algorithm, implemented in emulate method

        # Create job
        job = ising_problem.sqa_job(gamma_t=gamma_t, tmax=annealing_time, nbshots=self.n_shots)
        
        # Calculate number of necessary steps to cover the annealing time        
        n_steps = int(np.floor(1 / self.delta_t))  
        
        # Define the SQA QPU parameters and measure time
        sqa_qpu = SQAQPU(temp_t=temp_t, n_steps=n_steps, n_trotters=int(n_trotters))
        
        # Submit and run the job
        result = sqa_qpu.submit(job)

        # Extract all solutions (in integer format and its probability)
        states = [(sample.state.int, sample.probability) for sample in result.raw_data]

        # Initialize variables to track the minimum energy
        min_energy = float("inf")
        for state in states:
            # Get the spin representation of the solution
            solution_configuration = integer_to_spins(state[0], self.n_spins)
            # Calculate the energy of the solution  
            energy = self.calculate_energy(self.J, self.h, solution_configuration)
            if energy < min_energy:
                min_energy = energy
        
        return min_energy



    def calculate_energy(self, J, h, spins):
        '''
        This function computes the energy of a spin configuration 

        Parameters
        ----------
        J: np.ndarray
            Symmetric matrix of dimension n_spins x n_spins that contains the couplings between spins
        
        h: np.ndarray
            Vector of dimension n_spins containing the local magnetic fields
        
        spins: list (ASK TO MAKE SURE)
            List of the spin values of each node

        Returns
        -------
        energy: float
            Energy of the spins configuration

        '''
        
        energy = 0
        for i in range(self.n_spins):
            energy += h[i] * spins[i]
        for i in range(self.n_spins):
            for j in range(i+1, self.n_spins):
                energy += J[i,j] * spins[i] * spins[j]
        return energy
    
    def generate_random_problems(self, n_examples=1, seed=42):
        '''
        Generates n_examples random coupling J matrices and local fields h vectors

        Parameters
        -----------
        n_examples: int
            Number of examples used for ML algorithm

        Returns
        --------
        J: np.ndarray
            Symmetric matrix of dimension n_spins x n_spins that contains the couplings between spins

        h: np.ndarray
            Vector of dimension n_spins containing the local magnetic fields

        X: np.ndarray
            Result of flattening J and concatenating it with h

        '''

        J = np.zeros((n_examples, self.n_spins, self.n_spins))
        h = np.zeros((n_examples, self.n_spins))

        random.seed(seed)  # Set the seed for reproducibility

        for i in range(n_examples):
            for j in range(self.n_spins):
                h[i,j] = random.uniform(0, 1)
                for k in range(j+1, self.n_spins):
                    J[i,j,k] = random.uniform(-1, 1)
                    J[i,k,j] = J[i,j,k]

        if n_examples == 1:
            J = J[0]
            h = h[0]

        self.J = J
        self.h = h

        return J, h

    
    def ising_params_from_graph(self, G):
        '''
        Converts a graph to an Ising problem, obtaining the coupling matrix J
        and the vector of local magnetic fields h

        Parameters:
        -----------
        G : networkx.Graph
            Graph with:
            - node attributes: 'h' (float), default 0
            - edge attributes: 'J' (float), default 0

        Returns:
        --------
        J: np.ndarray
            Symmetric matrix of dimension n_spins x n_spins that contains the couplings between spins

        h: np.ndarray
            Vector of dimension n_spins containing the local magnetic fields
        '''

        n = G.number_of_nodes()
        self.h = np.zeros(n)
        self.J = np.zeros((n, n))

        node_to_index = {node: idx for idx, node in enumerate(G.nodes)} # Creates a dictionary with {"0":0, ...}

        for node, idx in node_to_index.items():
            self.h[idx] = G.nodes[node].get("h", 0.0)

        for u, v, data in G.edges(data=True):
            i, j = node_to_index[u], node_to_index[v]
            self.J[i, j] = data.get("J", 0.0)
            self.J[j, i] = self.J[i, j]  # symmetric

        return self.J, self.h
    
    def load_graph_from_file(self, filepath):
        """
        Loads a NetworkX graph from a pickle (.pkl) file.

        Parameters:
        - filepath: full path to the .pkl file.

        Returns:
        - G: the loaded NetworkX graph.
        """
        # Load the graph using pickle
        with open(filepath, 'rb') as f:
            G = pickle.load(f)

        return G


    def unflatten_X(self, x_vector):
        # Number of elements in upper triangle without diagonal
        n_upper = self.n_spins * (self.n_spins - 1) // 2

        # Split vector into J_upper and h parts
        J_upper = x_vector[:n_upper]
        h = x_vector[n_upper:]

        # Initialize full J matrix
        J = np.zeros((self.n_spins, self.n_spins))

        # Indices for upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(self.n_spins, k=1)

        # Fill upper triangle
        J[triu_indices] = J_upper

        # Make symmetric by copying upper to lower triangle
        J = J + J.T

        return J, h

    def calculate_exact_gs_energy(self):
        '''
        Transform an Ising problem defined by J and h into a matrix form
        '''
        n = self.n_spins
        
        obs_h = Observable(n, pauli_terms=[Term(self.h[i], "Z", [i]) for i in range(n)])
        obs_h_matrix = obs_h.to_matrix()
        obs_J = Observable(n, pauli_terms=[Term(self.J[i, j], "ZZ", [i, j]) for i in range(n) for j in range(i + 1, n)])
        obs_J_matrix = obs_J.to_matrix()
        hamiltonian_matrix = obs_h_matrix + obs_J_matrix
        eigenvalues, _ = np.linalg.eigh(hamiltonian_matrix)
        ground_state_energy = eigenvalues[0]

        return ground_state_energy



############################################################################
############## ML_hyperparameter_tuning (Bayesian Optimization) ############
############################################################################

class BayesianOptimizer:
    def __init__(self, sqa, bounds, n_init=5, kernel=None, xi=0.01, seed=42):
        """
        Parameters:
            func : class
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

        self.func = sqa.solve
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.xi = xi
        self.kernel = kernel if kernel else Matern(nu=2.5)
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True)

        # Initialize with random samples
        self.X_sample = self._random_sample(n_init)
        self.Y_sample = self._evaluate(self.X_sample)

    def _random_sample(self, n):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n, self.dim))

    def _evaluate(self, X):
        return np.array([self.func(x) for x in X]).reshape(-1, 1)

    def _expected_improvement(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu_sample_opt = np.min(self.Y_sample)
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    

    def suggest(self, n_candidates=1000):
        X_grid = self._random_sample(n_candidates)
        ei = self._expected_improvement(X_grid)
        return X_grid[np.argmax(ei)]

    def step(self):
        self.model.fit(self.X_sample, self.Y_sample)
        x_next = self.suggest().reshape(1, -1)
        y_next = self._evaluate(x_next)
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


#################################################################
################## Brute Force Solver ###########################
#################################################################

import numpy as np
from itertools import product

class Brute_Force_Solver:
    def __init__(self, J: np.ndarray, h: np.ndarray):
        """
        Initialize the Ising problem.
        
        Parameters:
        - J: numpy.ndarray, symmetric interaction matrix (n x n)
        - h: numpy.ndarray, local field vector (length n)
        """
        self.J = np.array(J)
        self.h = np.array(h)
        self.n = len(h)
        self._validate_inputs()
    
    def _validate_inputs(self):
        assert self.J.shape == (self.n, self.n), "J must be a square matrix matching the size of h"
        assert np.allclose(self.J, self.J.T), "J must be symmetric"
    
    def energy(self, spins):
        """
        Compute Ising energy for a given spin configuration.
        
        Parameters:
        - spins: list or np.ndarray of -1/+1 values
        
        Returns:
        - energy: float
        """
        spins = np.array(spins)
        interaction = np.sum(np.triu(self.J * np.outer(spins, spins), k=1))
        field = np.dot(self.h, spins)
        return interaction + field

    def brute_force_solve(self):
        """
        Find the ground state energy and spin configuration using brute force.

        Returns:
        - min_energy: float
        - best_config: tuple of spin values (-1 or +1)
        """
        min_energy = float('inf')
        best_config = None
        
        for config in product([-1, 1], repeat=self.n):
            energy = self.energy(config)
            if energy < min_energy:
                min_energy = energy
                best_config = config

        return min_energy, best_config
