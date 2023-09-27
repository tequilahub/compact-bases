"""
example on how to compute the MRSQK results in the paper
see https://github.com/evangelistalab/qforte for qforte installation
"""

import os
from qforte import Circuit, build_circuit, QubitOperator, Molecule, MRSQK, SRQK, NTSRQK, system_factory
import tequila as tq
import numpy as np

fci=-2.012674126630611

geom = [('H', (0., 0., 0.0)), ('H', (0., 0., 1.50)), ('H', (0., 0., 3.00)), ('H', (0., 0., 4.50))]
mol = system_factory(build_type='psi4', mol_geometry=geom, basis='sto-6g')

s=1
d=1
ref = [1,1,1,1,0,0,0,0]

result = {}
for m in [1,8]:
    mrsqk = MRSQK(mol, reference=ref, trotter_number=m)
    mrsqk.run(s=s, d=d, dt_o=0.8)
    result[(d,s,m)] = mrsqk.get_gs_energy()

print(fci)
print(result.items())
for k,v in result.items():
    print("error for {} : {:2.5f}".format(k,fci-v))
