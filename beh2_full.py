import tequila as tq
import numpy
from src.utils import Rot, Corr, GNM, gem_fast
import time
start = time.time()

# printout original script
# G(1,1) - molecular: 0.019315171507825468
# G(1,1) - atomic:  0.019866583529328352
# G(2,2)+UR:  -0.0013985236433935455
# took 84.57138419151306s  should take s 

# Compute PES points of BeH2 with the G(2,2) method and others
# Same as in https://arxiv.org/abs/2302.10660
# here with px and py not frozen

# add an additional double excitation between px and py
add_px_py_ex = True

data1 = {}
data2 = {}

# define the molecule
geometry1 = """
Be 0.0 0.0 0.0
H 0.0 0.0 2.5
H 0.0 0.0 -2.0
"""

# px and py orbitals have indices: 3 and 4
mol = tq.Molecule(geometry=geometry1, basis_set="sto-6g")
# exact energy in the fiven basis set
fci = mol.compute_energy("fci")

# SPA wavefunction for the molecular graph
# px and py are assigned to left and right bond -- could be any way 
U0 = mol.make_ansatz(name="SPA", edges=[(0,2,3),(1,5,4)])
# adding orbial rotations
# use mol.print_basis_info() to see coefficients of molecular orbitals
# combining orbital 0 and 1 gives orbitals that roughly look like the left and right bond. The same for 2 and 5 with anti-bonding orbitlas
U0+= mol.UR(0,1, (tq.Variable((0,1,"0"))+0.5)*numpy.pi)
U0+= mol.UR(2,5, (tq.Variable((2,5,"0"))+0.5)*numpy.pi)
if add_px_py_ex:
    U0+= mol.UC(3,4,"C-pX-pY")
# adding relaxation (the "+ UR")
U0+= mol.UR(0,2,"a")
U0+= mol.UR(0,5,"b")
U0+= mol.UR(1,2,"c")
U0+= mol.UR(1,5,"d")
# optimize
H = mol.make_hamiltonian()
E = tq.ExpectationValue(H=H, U=U0)
result0 = tq.minimize(E, silent=True)
print("G(1,1) - molecular:", result0.energy - fci)

# now the same for the atomic graph, 
# px and py both on Be (naturally)

# nachbau
U1 = mol.make_ansatz(name="SPA", edges=[(0,1,3,4),(2,5)], label="1")
# the orbital rotations lead to roughly atomic orbitals
U1+= mol.UR(0,2, (tq.Variable((0,1,"1"))+0.5)*numpy.pi)
U1+= mol.UR(1,5, (tq.Variable((1,5,"1"))+0.5)*numpy.pi)
# adding relaxation
U1+= mol.UR(0,1,"aa")
U1+= mol.UR(0,5,"bb")
U1+= mol.UR(1,0,"cc")
U1+= mol.UR(1,5,"dd")
U1+= mol.UR(1,2,"ee")
U1+= mol.UR(2,5,"gg")
E = tq.ExpectationValue(H=H, U=U1)
result1 = tq.minimize(E, silent=True)
print("G(1,1) - atomic: ", result1.energy - fci)

# save initial variables
variables = {**result0.variables, **result1.variables}

# run G(2,2)+UR
v,vv,variables = GNM(circuits=[U0,U1], variables=variables, H=H, silent=False, M=2)
print("G(2,2)+UR: ", fci-v[0])
end = time.time()
print("took {}s".format(end-start), " should take <2min on intel i5-12500" 
