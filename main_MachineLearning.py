import time
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from classical_shadow import *

class ClassicalShadowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim, self.hidden_layers = hidden_dim, hidden_layers
        if hidden_layers == 0:
            net = [nn.Linear(input_dim, output_dim), nn.ReLU()]
        else:
            net = []
            net.append(nn.Linear(input_dim, hidden_dim))
            net.append(nn.ReLU())
            for _ in range(hidden_layers):
                net.append(nn.Linear(hidden_dim, hidden_dim))
                net.append(nn.ReLU())
            net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
    def forward(self, x):
        y = self.net(x)
        y = y.reshape(-1, self.output_dim//6, 6)
        y = nn.Softmax(dim=-1)(y)
        return y

class QuantumCircuitData(Dataset):
    def __init__(self, num_data, num_layers, num_qubits, num_shots, use_true_dm):
        x, y = [], []
        for random_seed in range(num_data):
            quantum_circuit = RandomRotation(num_layers, num_qubits, num_shots, random_seed)
            x.append(torch.FloatTensor(quantum_circuit.params.reshape(-1)))
            if use_true_dm:
                y.append(torch.from_numpy(quantum_circuit.density_matrix))
            else:
                y.append(torch.from_numpy(quantum_circuit._reconstruct_classical_shadow()))
        self.x, self.y = x, y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.y)

num_train  = 10000
num_test   = 100
num_layers = 1
num_qubits = int(input("num_qubits = "))
num_shots  = int(input("num_shots = "))

input_dim     = num_layers * num_qubits * 3 # each qml.Rot has 3 parameters
output_dim    = num_shots * num_qubits * 6 # 6 different basis
hidden_layers = 1
hidden_dim    = num_shots * num_qubits

num_epochs   = 50
train_data   = QuantumCircuitData(num_train, num_layers, num_qubits, num_shots=100, use_true_dm=False)
test_data    = QuantumCircuitData(num_test, num_layers, num_qubits, num_shots=100, use_true_dm=True)
train_loader = DataLoader(train_data, min(16, num_train//10), shuffle=True, drop_last=False)
test_loader  = DataLoader(test_data, min(16, num_test//10), shuffle=False, drop_last=False)

def classical_shadow_transform(y):
    single_qubit_eigenstates = [
        (1/np.sqrt(2))*np.array([[1],[1]]), 
        (1/np.sqrt(2))*np.array([[1],[-1]]),
        (1/np.sqrt(2))*np.array([[1j],[1]]), 
        (1/np.sqrt(2))*np.array([[-1j],[1]]),
        np.array([[1],[0]]), 
        np.array([[0],[1]]),
    ]
    density_matrix_map = lambda x: 3 * x @ np.conjugate(x.T) - np.eye(2)
    single_qubit_density_matrix = torch.from_numpy(
        np.array(list(map(density_matrix_map, single_qubit_eigenstates)))
    )
    num_shots, num_qubits = y.shape[0], y.shape[1]
    classical_shadow_density_matrix = torch.zeros(2**num_qubits, 2**num_qubits, dtype=torch.cdouble)
    for s in range(num_shots):
        single_shot_density_matrix = torch.ones(1)
        for q in range(num_qubits):
            # composite_single_density_matrix = torch.max(y[s][q]) * single_qubit_density_matrix[torch.argmax(y[s][q])]
            composite_single_density_matrix = torch.sum(y[s][q][:, None, None] * single_qubit_density_matrix, dim=0)
            single_shot_density_matrix = torch.kron(single_shot_density_matrix, composite_single_density_matrix)
        classical_shadow_density_matrix += single_shot_density_matrix
    classical_shadow_density_matrix /= num_shots
    return classical_shadow_density_matrix

def HS_loss(A, B):
    return -torch.abs(torch.trace(torch.adjoint(A) @ B))

def random_operator_loss(A, B, num_qubits, samples=10):
    mean_loss = 0
    for _ in range(samples):
        measurements = [
            torch.tensor([[0,1],[1,0]], dtype=torch.cdouble),
            torch.tensor([[0,-1j],[1j,0]], dtype=torch.cdouble),
            torch.tensor([[1,0],[0,-1]], dtype=torch.cdouble),
        ]
        observable = torch.ones(1)
        for q in range(num_qubits):
            observable = torch.kron(observable, np.random.choice(measurements))
        mean_loss += abs(torch.trace(A@observable) - torch.trace(B@observable))
    return mean_loss / samples

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3, weight_decay=1E-3)
    record = np.zeros((4, num_epochs))
    for epoch in range(num_epochs):
        model.train()
        train_HS_loss, train_meas_loss = 0, 0
        for x, y_true in train_loader:
            loss = 0
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.reshape(-1, num_shots, num_qubits, 6)
            for i in range(len(y_pred)):
                y_cs = classical_shadow_transform(y_pred[i])
                loss += HS_loss(y_cs, y_true[i])
                train_HS_loss += loss.detach()
                train_meas_loss += random_operator_loss(y_cs, y_true[i], num_qubits).detach()
            loss /= len(x)
            loss.backward()
            optimizer.step()
        train_HS_loss /= num_train
        train_meas_loss /= num_train
        
        model.eval()
        test_HS_loss, test_meas_loss = 0, 0
        for x, y_true in test_loader:
            y_pred = model(x)
            y_pred = y_pred.reshape(-1, num_shots, num_qubits, 6)
            for i in range(len(y_pred)):
                y_cs = classical_shadow_transform(y_pred[i])
                test_HS_loss += HS_loss(y_cs, y_true[i]).detach()
                test_meas_loss += random_operator_loss(y_cs, y_true[i], num_qubits).detach()
        test_HS_loss /= num_test
        test_meas_loss /= num_test
        record[0][epoch], record[1][epoch] = train_HS_loss.item(), test_HS_loss.item()
        record[2][epoch], record[3][epoch] = train_meas_loss.item(), test_meas_loss.item()
        print(epoch, train_HS_loss.item(), test_HS_loss.item(), train_meas_loss.item(), test_meas_loss.item())
    return record

model  = ClassicalShadowModel(input_dim, hidden_dim, hidden_layers, output_dim)
record = train(model)
np.save(f"result_ml_l{num_layers}_q{num_qubits}_s{num_shots}", record)