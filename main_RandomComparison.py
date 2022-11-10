import time
from itertools import product
from classical_shadow import *

loop_seed    = list(range(25))
loop_layers  = [1,2]
loop_qubits  = [2,3,4,5]
loop_shots   = [int(1E2), int(1E3), int(1E4), int(1E5)]
loop_product = product(loop_seed, loop_layers, loop_qubits, loop_shots)

result_dict = {}
t_total_start = time.time()
for random_seed, num_layers, num_qubits, num_shots in loop_product:
    test_circuit = RandomRotation(num_layers, num_qubits, num_shots, random_seed)
    result_key = f"seed{random_seed}_l{num_layers}_q{num_qubits}_shots{num_shots}"
    print(f"start {result_key} ... ", end="")
    t_start = time.time()
    result_dict[result_key] = [
        test_circuit.get_HS_norm(random_construct=False),
        test_circuit.get_HS_norm(random_construct=True),
        test_circuit.get_Frob_norm(random_construct=False),
        test_circuit.get_Frob_norm(random_construct=True),
    ]
    t_end = time.time()
    print(f"finish {result_key} | t = {t_end-t_start:.2f}")
t_total_end = time.time()
print(f"Total time cost = {t_total_end - t_total_start:.2f}")

np.save("result_random_comparison.npy", result_dict, allow_pickle=True)