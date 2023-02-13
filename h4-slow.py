"""
H4 Ring example from the paper
Slower implementation using source code given in the paper
Recommending to use h4.py (sigificantly faster) and wait until the functionalty is taken over into tequila
Needs tequila version >= 1.8.5
"""
import tequila as tq
import numpy
import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)

def G(N,M,H,U,initial_values):
    print(N," ",M)
    assert M <= N
    assert N <= len(U)

    # initial values for coefficients through GEM (Eq.1)
    v,vv = tq.apps.gem(U, H, variables=initial_values)
    
    # initialize variables for coefficients
    c = [tq.Variable(("c",i)) for i in range(N)]
    
    for i in range(N):
        initial_values[c[i]]=vv[i,0]

    # Initialize the objective of Eq.7 
    # by expanding with wavefunction of Eq.6
    ED = 0.0
    EN = 0.0
    for i in range(N):
        for j in range(N):
            # only need real part of braket (by construction)
            EN += c[i]*c[j]*tq.braket(U[i],U[j],H)[0]
            ED += c[i]*c[j]*tq.braket(U[i],U[j])[0]
    
    E = EN/ED
    
    # active variables for the concerted optimization
    active = sum([U[i].extract_variables() for i in range(M)],[])
    active += c

    # gradient compilation not optimized for tq.braket
    # recommending to use finite-differences
    result = tq.minimize(E, variables=active, initial_values=initial_values, gradient="2-point", method_options={"finite_diff_rel_step":1.e-4})
    
    return result

# define the molecule
geometry = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
mol = tq.Molecule(geometry=geometry, basis_set="sto-6g")
# use orthonormalized atomic orbitals
mol = mol.use_native_orbitals()

# Define the qubit Hamiltonian
H = mol.make_hamiltonian()
# get the reference energy
exact = mol.compute_energy("fci")

# define the graphs
G1 = [(0,1),(2,3)]
G2 = [(0,2),(1,3)]
G3 = [(0,3),(1,2)]
graphs = [G1, G2, G3]

# define the circuits to generate the basis
circuits = []
for i,edges in enumerate(graphs):
    # label to prevent identical variable names in different circuits
    label="G{}".format(i+1)
    U = mol.make_ansatz(name="SPA", edges=edges, label=label)
    for e in edges:
        i,j = mol.format_excitation_indices(((e[0],e[1]),))[0]
        U += mol.UR(e[0],e[1],label=label)
    circuits.append(U)

# pre-optimize the circuits
variables = {}
for U in circuits:
    E = tq.ExpectationValue(U=U, H=H)
    result = tq.minimize(E, silent=True)
    variables = {**variables, **result.variables}
    print("Graph {}: with energy {:+2.5f}".format(i, result.energy))

# run the G(N,M) computations
energies={}
for N in range(1,4):
    for M in range(1,N):
        result = G(N,M,H,circuits,variables)
        variables = result.variables
        energies["G({},{})".format(N,M)]=result.energy
        print("G({},{}) with energy {:+2.5f}".format(N,M,result.energy))

print(energies)


