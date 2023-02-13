import scipy
import qulacs
import tequila as tq
import numpy
import scipy

import warnings
warnings.filterwarnings("ignore", category=tq.TequilaWarning)

"""
Convenience Implementations and Structures to speed up simulation times
"""

def Rot(idx, mol, label=None, s=0.5):
    """
    Convenience implementation of Rotation gates as described in the paper
    See also ArXiv:2207.12421 Eq.(6)
    In tequila version >= 1.8.4 this is equivalent to mol.UR
    """
    angle=tq.Variable((tuple(idx),label))
    tmp = mol.make_excitation_gate(indices=[(2*idx[0],2*idx[1])], angle=(angle+s)*numpy.pi)
    tmp+= mol.make_excitation_gate(indices=[(2*idx[0]+1,2*idx[1]+1)], angle=(angle+s)*numpy.pi)
    return tmp

def Corr(i,j, label=None):
    """
    Convenience initialization of paired two-body correlator
    See ArXiv:2207.12421 Eq.(22)
    In tequila version >= 1.8.4 this is equivalent to mol.UC
    """
    return tq.gates.QubitExcitation(target=[2*i,2*j,2*i+1,2*j+1], angle=(i,j,label))

class BraKetQulacs:
    """
    Hacky Replacement of tq.BraKet
    Speedup of underlying simulation
    Limitations: Only Qulacs can be backend, not differentiable at the moment
    """
    def __init__(self, bra,ket,H):
        # translate tq -> qulacs
        E1 = tq.compile(tq.ExpectationValue(U=bra,H=H), backend="qulacs")
        E2 = tq.compile(tq.ExpectationValue(U=ket,H=H), backend="qulacs")
        # extract qulacs structures
        self.bra = E1.get_expectationvalues()[0]._U
        self.ket = E2.get_expectationvalues()[0]._U
        self.H = E1.get_expectationvalues()[0]._H[0]
        self.n_qubits = ket.n_qubits
        self.is_overlap = H.n_qubits == 0
    def __call__(self, variables, *args, **kwargs):
        # call qulacs structures
        # similar as tequila would, but exploits storing wavefunctions
        self.ket.update_variables(variables)
        self.bra.update_variables(variables)
        state_bra = self.bra.initialize_state(self.n_qubits)
        state_ket = self.ket.initialize_state(self.n_qubits)
        self.bra.circuit.update_quantum_state(state_bra)
        self.ket.circuit.update_quantum_state(state_ket)
        if self.is_overlap:
            vector1 = state_bra.get_vector()
            vector2 = state_ket.get_vector()
            result = vector1.T.dot(vector2)
        else:
            result = self.H.get_transition_amplitude(state_bra, state_ket)

        result=result.real
        return result


def gem_fast(circuits, H, variables=None):
    """
    Fast implementation of tq.apps.gem 
    works only with qulacs backend
    not differentiable
    """
    E = [tq.simulate(tq.ExpectationValue(H=H, U=U), variables=variables) for U in circuits]
    SS = numpy.eye(len(circuits))
    EE = numpy.eye(len(circuits))
    for i in range(len(circuits)):
        EE[i,i] = E[i]
        for j in range(i+1,len(circuits)):
            f=BraKetQulacs(circuits[i], circuits[j], H)
            ff=BraKetQulacs(circuits[i], circuits[j], H=tq.paulis.I())
            EE[i,j] = f(variables)
            EE[j,i] = EE[i,j]
            SS[i,j] = ff(variables)
            SS[j,i] = SS[i,j]
    v,vv = scipy.linalg.eigh(EE,SS)
    return v,vv

class BigExpVal:
    """
    Convenience to initialize an expectation value as described in Eq.(7) of the paper with the Qulacs only structure
    """

    def __init__(self, circuits, H, coeffs):
        n = len(circuits)
        self.n = n
        E = [tq.compile(tq.ExpectationValue(H=H, U=U)) for U in circuits]
        SS = []
        EE = []
        for i in range(n):
            tmp1 = []
            tmp2 = []
            for j in range(i):
                xEE=BraKetQulacs(circuits[i],circuits[j],H=H)
                xSS=BraKetQulacs(circuits[i],circuits[j],H=tq.paulis.I())
                tmp1.append(xEE)
                tmp2.append(xSS)
            tmp1.append(E[i])
            tmp2.append(1.0)
            EE.append(tmp1)
            SS.append(tmp2)
        self.SS = SS
        self.EE = EE
        self.coeffs = coeffs
        variables={}
        for U in circuits:
            variables = {**variables, **{x:0.0 for x in U.extract_variables()}}
        for c in coeffs:
            variables = {**variables, **{x:0.0 for x in  c.extract_variables()}}
        self.variables=list(variables.keys())

    def __call__(self, x,*args, **kwargs):
        n = self.n
        assert len(x) <= len(self.variables)
        values={self.variables[i]:x[i] for i in range(len(self.variables))}
        c = [self.coeffs[i](values) for i in range(n)]
        f = 0.0
        s = 0.0
        for i in range(n):
            f+=self.EE[i][i](values)*c[i]**2
            s+=c[i]**2
            for j in range(i):
               f+=2.0*self.EE[i][j](values)*c[i]*c[j]
               s+=2.0*self.SS[i][j](values)*c[i]*c[j]

        f=f.real
        s=s.real
        if s>0.0:
            r=f/s
        else:
            # failsave for optimizer, only happens with bad variable initialization
            r=1e5
        return r

def GMN(circuits, H, variables, silent=False, maxiter=10, M=None):
    """
    the G(M,N) method from the paper, N is implicitly given over the number of circuits
    """
    N = len(circuits)
    if M is None:
        M = len(circuits)
    
    # fix variables for circuits that will not be part of the optimization
    for i in range(M, N):
        U = circuits[i]
        U = U.map_variables(variables)
        circuits[i] = U

    vkeys = []
    for U in circuits:
        vkeys+=U.extract_variables()
    
    variables = {**{k:0.0 for k in vkeys if k not in variables}, **variables}
   
    v,vv = gem_fast(circuits,H,variables)
    
    x0 = {k:variables[k] for k in vkeys}

    coeffs = []
    for i in range(len(circuits)):
        c=tq.Variable(("c",i))
        coeffs.append(c)
        x0[c] = vv[i,0]
        vkeys.append(c)
        
    energy = 1.0
    def callback(x):
        energy=f(x)
        if not silent:
            print("current energy: {:+2.4f}".format(energy))
    
    f = BigExpVal(circuits=circuits, H=H, coeffs=coeffs)

    for i in range(maxiter):
        result = scipy.optimize.minimize(f, x0=list(x0.values()), jac="2-point", method="bfgs", options={"finite_diff_rel_step":1.e-5, "disp":True}, callback=callback)
        x0 = {vkeys[i]:result.x[i] for i in range(len(result.x))}
        v,vv = static_krylov(circuits,H,x0)
        for i in range(len(coeffs)):
            x0[coeffs[i]]=vv[i,0]
        if numpy.isclose(energy, v[0], atol=1.e-4):
            print("not converged")
            print(energy)
            print(v[0])
            energy = v[0]
        else:
            energy = v[0]
            break
    
    for k in vkeys:
        variables[k] = x0[k]
    return v,vv,variables


        
    


