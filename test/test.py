import tequila as tq
import numpy as np
from qvalence import Rot, GNM, gem_fast, BigExpVal, BraKetQulacs
from tequila.objective.braket import make_overlap, make_transition
from tequila.tools.random_generators import make_random_hamiltonian

import pytest

def test_h4():
    # define the molecule
    geometry1 = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"

    mol = tq.Molecule(geometry=geometry1, basis_set="sto-6g")
    mol = mol.use_native_orbitals()
    H = mol.make_hamiltonian()
    # exact ground state energy
    fci = mol.compute_energy("fci")

    # define the graphs
    edges1 = [(0,1),(2,3)]
    edges2 = [(0,2),(1,3)]
    edges3 = [(0,3),(1,2)]
    graphs = [edges1, edges2, edges3]

    # define the circuits to generate the basis
    circuits = []
    rot_circuits = []
    spa_circuits = []
    wfns = []
    for i,edges in enumerate(graphs):
        U = mol.make_ansatz(name="SPA", edges=edges, label="G{}".format(i))
        spa_circuits.append(U)
        UR = tq.QCircuit()
        for e in edges:
            UR += Rot(e,mol,i) 
        rot_circuits.append(UR)
        circuits.append(U+UR)

    # pre-optimize the circuits
    variables_preopt = {}
    energies = []
    for U in circuits:
        E = tq.ExpectationValue(U=U, H=H)
        result = tq.minimize(E, silent=True)
        variables_preopt = {**variables_preopt, **result.variables}
        energies.append(result.energy)
        wfn = tq.simulate(U, variables=result.variables)
        wfns.append(wfn)
    variables = {**variables_preopt}

    v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
    
    assert np.isclose(fci, v[0])

def test_braket():

    # make random circuits
    # np.random.seed(111)
    n_qubits = np.random.randint(1, high=5)
    U = {k:tq.make_random_circuit(n_qubits) for k in range(2)}

    ######## Testing expectation value #########
    # make random hamiltonian
    paulis = ['X','Y','Z']
    n_ps = np.random.randint(1, high=2*n_qubits+1)
    H = make_random_hamiltonian(n_qubits, paulis=paulis, n_ps=n_ps)

    qulacs_exp_value = BraKetQulacs(bra=U[0], ket=U[0], H=H)
    br_exp_value_real = tq.braket(ket=U[0], operator=H)[0]
    br_exp_value = tq.simulate(br_exp_value_real)

    assert np.isclose(qulacs_exp_value({}), br_exp_value, atol=1.e-4)

    ######## Testing overlap #########
    qulacs_overlap = BraKetQulacs(ket=U[0], bra=U[1], H=tq.paulis.I())
    br_objective_real = tq.braket(ket=U[0], bra=U[1])[0]
    br_overlap = tq.simulate(br_objective_real)
    
    assert np.isclose(br_overlap, qulacs_overlap({}), atol=1.e-4)

    ######## Testing transition element #########
    qulacs_trans_el = BraKetQulacs(ket=U[0], bra=U[1], H=H)
    br_trans_real = tq.braket(ket=U[0], bra=U[1], operator=H)[0]
    br_trans_el = tq.simulate(br_trans_real)

    assert np.isclose(br_trans_el, qulacs_trans_el({}), atol=1.e-4)