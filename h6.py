# 7 Graphs
# good pre-optimization
# behaves quite nice

# dependencies
# pip install tequila-basic
# pip install pyscf
# optional but recommended
# pip install qulacs 
import tequila as tq
import numpy
from tequila.apps.krylov import krylov_method

import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)

from utils import Rot, gem_fast, BigExpVal, GNM, Corr 

# plot data:
error_g1=0.0
error_g1_g2=0.0
error_refined=0.0

# Initialize
geometry="h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5\nh 0.0 0.0 6.0\nh 0.0 0.0 7.5"
mol = tq.Molecule(geometry=geometry, basis_set="sto-6g")
fci = mol.compute_energy("fci")
print("fci : {:+2.4f}".format(fci))
mol = mol.orthonormalize_basis_orbitals()
H = mol.make_hamiltonian()

# Create SPA circuits
UG1 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,3),(4,5)], label="G1")
UG2 = mol.make_ansatz(name="SPA", edges=[(1,2),(3,4),(5,0)], label="G2")
UG3 = mol.make_ansatz(name="SPA", edges=[(1,2),(0,3),(4,5)], label="G3")
UG4 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,5),(3,4)], label="G4")
# less important
edges5=[(0,2),(1,3),(4,5)]
edges6=[(0,1),(2,4),(3,5)]
edges7=[(0,4),(2,3),(1,5)]
UG5 = mol.make_ansatz(name="SPA", edges=[(0,2),(1,3),(4,5)], label="G5")
UG6 = mol.make_ansatz(name="SPA", edges=[(0,1),(2,4),(3,5)], label="G6")
UG7 = mol.make_ansatz(name="SPA", edges=[(0,4),(2,3),(1,5)], label="G7")

U5 = UG5
for edge in edges5:
    U5 += Rot(edge,mol=mol,label="RG5")
U6 = UG6
for edge in edges6:
    U6 += Rot(edge,mol=mol,label="RG6")
U7 = UG7
for edge in edges7:
    U7 += Rot(edge,mol=mol,label="RG7")

# G1
UR1 = Rot((0,1),mol=mol,label="RG1")
UR1+= Rot((2,3),mol=mol,label="RG1")
UR1+= Rot((4,5),mol=mol,label="RG1")
# more freedom
UR1+= Rot((1,2),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((3,4),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((0,5),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((0,2),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((1,3),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((2,4),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((3,5),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((0,4),mol=mol,label="G1",s=1.e-3)
UR1+= Rot((1,5),mol=mol,label="G1",s=1.e-3)


# G2
UR2 = Rot((1,2),mol=mol,label="G2")
UR2+= Rot((3,4),mol=mol,label="G2")
UR2+= Rot((0,5),mol=mol,label="G2")
# more freedom
UR2+= Rot((0,1),mol=mol,label="G2x",s=1.e-3)
UR2+= Rot((2,3),mol=mol,label="G2x",s=1.e-3)
UR2+= Rot((4,5),mol=mol,label="G2x",s=1.e-3)
UR2+= Rot((1,2),mol=mol,label="G2x",s=1.e-3)
UR2+= Rot((3,4),mol=mol,label="G2x",s=1.e-3)
UR2+= Rot((2,3),mol=mol,label="G2x",s=1.e-3)

#G 3
UR3 = Rot((1,2),mol=mol,label="G3")
UR3+= Rot((0,3),mol=mol,label="G3")
UR3+= Rot((4,5),mol=mol,label="G3")
# more freedom
UR3+= Rot((0,1),mol=mol,label="G3x",s=1.e-3)
UR3+= Rot((2,3),mol=mol,label="G3x",s=1.e-3)
UR3+= Rot((1,2),mol=mol,label="G3x",s=1.e-3)
UR3+= Rot((3,4),mol=mol,label="G3x",s=1.e-3)
UR3+= Rot((2,3),mol=mol,label="G3x",s=1.e-3)


#G 4
UR4 = Rot((0,1),mol=mol,label="G4")
UR4+= Rot((2,5),mol=mol,label="G4")
UR4+= Rot((3,4),mol=mol,label="G4")
# more freedom
UR4+= Rot((2,3),mol=mol,label="G4x",s=1.e-3)
UR4+= Rot((4,5),mol=mol,label="G4x",s=1.e-3)
UR4+= Rot((1,2),mol=mol,label="G4x",s=1.e-3)
UR4+= Rot((3,4),mol=mol,label="G4x",s=1.e-3)
UR4+= Rot((2,3),mol=mol,label="G4x",s=1.e-3)


U1 = UG1 + UR1
U2 = UG2 + UR2
U3 = UG3 + UR3
U4 = UG4 + UR4

variables = {}
for i,U in enumerate([U1,U2,U3,U4,U5,U6,U7]):
    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    print("Error Graph {} alone = {:2.5f}".format(i, result.energy-fci))
    variables = {**variables, **result.variables}

v,vv = gem_fast(circuits=[U1,U2], H=H, variables=variables)
print("Error G(2,0): {:2.5f}".format(v[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4], H=H, variables=variables)
print("Error G(4,0): {:2.5f}".format(vx[0]-fci))

vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6], H=H, variables=variables)
print("Error G(6,0): {:2.5f}".format(vx[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6,U7], H=H, variables=variables)
print("Error G(7,0): {:2.5f}".format(vx[0]-fci))

print("optimizing G(2,2):")

#add more rotation freedom
#U1x=mol.make_ansatz(name="GS", include_reference=False, label="G1-GS")
#U2x=mol.make_ansatz(name="GS", include_reference=False, label="G2-GS")
#U1+=U1x
#U2+=U2x
#variables = {**variables, **{k:1.e-3 for k in U1x.extract_variables() + U2x.extract_variables()}}

v,vv, variables = GNM(circuits=[U1,U2], H=H, variables=variables)
print("Error G(2,2): {:2.5f}".format(v[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4], H=H, variables=variables)
print("Error G(4,2): {:2.5f}".format(vx[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6], H=H, variables=variables)
print("Error G(6,2): {:2.5f}".format(vx[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6,U7], H=H, variables=variables)
print("Error G(7,2): {:2.5f}".format(vx[0]-fci))

print("optimizing G(4,4):")
v,vv, variables = GNM(circuits=[U1,U2,U3,U4], H=H, variables=variables)
print("Error G(4,4): {:2.5f}".format(v[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6], H=H, variables=variables)
print("Error G(6,4): {:2.5f}".format(vx[0]-fci))
vx,vvx = gem_fast(circuits=[U1,U2,U3,U4,U5,U6,U7], H=H, variables=variables)
print("Error G(7,4): {:2.5f}".format(vx[0]-fci))

