import tequila as tq
import numpy
from qvalence import Rot, Corr, GNM, gem_fast

"""
Compute the data from Fig.1 in the paper
uses accelerated structures
should take ~30 seconds
"""

data1 = {}
data2 = {}

# define the molecule
# rectangular
geometry1 = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
# linear
geometry2 = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"

mol = tq.Molecule(geometry=geometry1, basis_set="sto-6g")
# replace with "orthonormalize_basis_orbitals()" for tq.version < 1.8.4
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

best = min(energies)

data1[(1,0)]=best
variables = {**variables_preopt}
# compute static energies with the pre-optimized basis
v,vv = gem_fast(circuits=circuits[:2], variables=variables, H=H)
data1[(2,0)]=v[0]
v,vv = gem_fast(circuits=circuits[:3], variables=variables, H=H)
data1[(3,0)]=v[0]

# relax circuit parameters
v,vv,variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True, M=1)
data1[(2,1)]=v[0]

v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=1)
data1[(3,1)]=v[0]

v,vv,variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True)
data1[(2,2)]=v[0]

v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=2)
data1[(3,2)]=v[0]

v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
data1[(3,3)]=v[0]

# add more freedeom in orbital rotations
# this is G(N,M)+UR
for i in range(len(circuits)):
    e0 = graphs[i][0] 
    e1 = graphs[i][1]
    UR = Rot((e0[0],e1[0]),mol,i)
    UR+= Rot((e0[1],e1[1]),mol,i)
    UR+= Rot((e0[0],e1[1]),mol,i)
    circuits[i] += UR

# reset variables to pre-opt
variables=variables_preopt

# relax circuit parameters
v,vv,variables = GNM(circuits=circuits[:1], variables=variables, H=H, silent=True, M=1)
data2[(1,0)]=v[0]
v,vv,variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True, M=1)
data2[(2,1)]=v[0]
v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=1)
data2[(3,1)]=v[0]

v,vv,variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True)
data2[(2,2)]=v[0]

v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=2)
data2[(3,2)]=v[0]
v,vv,variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
data2[(3,3)]=v[0]

print("\n\nFinished!\n\n")

print("G(N,M) errors:")
for k,v in data1.items():
    error = abs(fci-v)
    print("G({},{}): {:+2.5f}".format(k[0],k[1],error))
print("G(N,M)+UR errors:")
for k,v in data2.items():
    error = abs(fci-v)
    print("G({},{})+UR: {:+2.5f}".format(k[0],k[1],error))



