import pennylane as qml
from pennylane import numpy as np

class ClassicalShadow:
    def __init__(self, num_qubits, num_shots):
        self.num_qubits = num_qubits
        self.num_shots  = num_shots
        self.device_c   = qml.device("default.qubit", wires=num_qubits, shots=num_shots)
        self.device_q   = qml.device("default.qubit", wires=num_qubits, shots=None)

        self.classical_shadow = None
        self.density_matrix   = None

    def _reconstruct_classical_shadow(self, random_construct=False):
        bits, recipes = self.classical_shadow
        classical_shadow_density_matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)

        single_qubit_eigenstates = [
            [(1/np.sqrt(2))*np.array([[1],[1]]), (1/np.sqrt(2))*np.array([[1],[-1]])],
            [(1/np.sqrt(2))*np.array([[1j],[1]]), (1/np.sqrt(2))*np.array([[-1j],[1]])],
            [np.array([[1],[0]]), np.array([[0],[1]])]
        ]
        density_matrix_map = lambda x: 3 * x @ np.conjugate(x.T) - np.eye(2)
        single_qubit_density_matrix = [list(map(density_matrix_map, es)) for es in single_qubit_eigenstates]

        for s in range(self.num_shots):
            single_shot_density_matrix = 1
            for q in range(self.num_qubits):
                if random_construct:
                    i, j = np.random.randint(3), np.random.randint(2)
                else:
                    i, j = recipes[s][q], bits[s][q]
                single_shot_density_matrix = np.kron(single_shot_density_matrix, single_qubit_density_matrix[i][j])
            classical_shadow_density_matrix += single_shot_density_matrix
        classical_shadow_density_matrix /= self.num_shots
        return classical_shadow_density_matrix

    def get_HS_norm(self, random_construct=False):
        # Hilbert-Schmidt inner product
        classical_shadow_density_matrix = self._reconstruct_classical_shadow(random_construct)
        return np.abs(np.trace(np.conjugate(classical_shadow_density_matrix.T) @ self.density_matrix))
    
    def get_Frob_norm(self, random_construct=False):
        # Frobenius norm
        classical_shadow_density_matrix = self._reconstruct_classical_shadow(random_construct)
        return np.linalg.norm(classical_shadow_density_matrix - self.density_matrix, ord="fro")

class RandomRotation(ClassicalShadow):
    def __init__(self, num_layers, num_qubits, num_shots, random_seed=0):
        super().__init__(num_qubits, num_shots)
        np.random.seed(random_seed)
        self.num_layers = num_layers
        self.params = np.random.rand(num_layers, num_qubits, 3)

        self.density_matrix   = qml.QNode(self._density_matrix, self.device_q)()
        self.classical_shadow = qml.QNode(self._classical_shadow, self.device_c)()
    def circuit(self):
        for l in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.Rot(*self.params[l][q], wires=q)
            for q in range(self.num_qubits):
                qml.CNOT(wires=[q, (q+1)%self.num_qubits])
    def _classical_shadow(self):
        self.circuit()
        return qml.classical_shadow(wires=range(self.num_qubits))
    def _density_matrix(self):
        self.circuit()
        return qml.density_matrix(wires=range(self.num_qubits))