import torch
from torch import nn
import numpy as np
from schrodinet.wavefunction.wf_base import WaveFunction
from schrodinet.wavefunction.rbf import RBF_Gaussian as RBF


from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient, Hessian
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit.circuit.library import RealAmplitudes 
from qiskit import Aer, transpile, assemble

from qiskit.utils.backend_utils import is_aer_provider, is_statevector_backend
from typing import Optional, List, Union, Dict, Sequence
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.register import Register
from qiskit.circuit.bit import Bit
from qiskit.quantum_info.operators import Operator
import numpy as np
from copy import deepcopy

from qiskit.opflow import (
    Z,
    X,
    I,
    StateFn,
    OperatorBase,
    TensoredOp,
    ExpectationBase,
    OperatorStateFn,
    CircuitSampler,
    ListOp,
    ExpectationFactory,
)

from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)

from qiskit.algorithms.optimizers import COBYLA, ADAM,CG
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import CXGate, UGate,U3Gate 

from qiskit import QuantumRegister
from qiskit.circuit import ParameterVector 
from qiskit.dagcircuit import DAGCircuit 

class ShiftOperator(QuantumCircuit):

    def __init__(
        self,
        regs: Union[Register, int, Sequence[Bit]],
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
        use_mct_ancilla: bool = False
    ):

        self.qreg = QuantumRegister(regs)
        super().__init__(self.qreg)
        

        if not use_mct_ancilla:
            for i in reversed(range(1, self.num_qubits)):
                self.mct(self.qreg[:i], self.qreg[i])
            self.x(self.qreg[0])
        else:
            qreg_shift_ancilla = QuantumRegister(self.num_qubits-3, 'q_shift_ancilla')
            self.add_register(qreg_shift_ancilla)
            for i in reversed(range(1, self.num_qubits)):
                self.mct(self.qreg[:i], self.qreg[i], qreg_shift_ancilla, mode='v-chain')
            self.x(self.qreg[0])


def GaussianAnsatz(num_qubits, domain, ipot=0, parameters='replace'):

    x = np.linspace(domain['xmin'],domain['xmax'],2**num_qubits)
   
    if ipot == 0:
        y = np.exp(-1./2*x**2).astype('float64')
    elif ipot == 1:
        x = x + 2
        y = np.exp(-np.exp(-x)-0.5*x)
    elif ipot == 2:
        y = np.exp(-np.abs(x-2.5)) + np.exp(-np.abs(x+2.5))

    # y = np.sqrt(y)
    y = y / np.linalg.norm(y)

    q = QuantumRegister(num_qubits,'q')
    qc = QuantumCircuit(q)
    qc.prepare_state(y)

    if parameters=='replace':
        dag = circuit_to_dag(qc.decompose(reps=10))

        P = ParameterVector('P',num_qubits)


        k = 0
        for i, node in enumerate(dag.topological_op_nodes()):

            if isinstance(node.op, (UGate,U3Gate)):
                
                _qc = QuantumCircuit(1)
                _qc.u(P[k],0,0,0)
                mini_dag = circuit_to_dag(_qc)
                dag.substitute_node_with_dag(node, mini_dag)
                k += 1
            if k == num_qubits:
                break
        return dag_to_circuit(dag)

    elif parameters=='add':
        dag = circuit_to_dag(qc.decompose(reps=10))
        P = ParameterVector('P',num_qubits)
        
        for i in range(num_qubits):
            dag.apply_operation_front(UGate(P[i],0,0), qargs=[q[i]])
        return dag_to_circuit(dag)
    else:

        return qc

class MultiQBitWaveFunction(WaveFunction):

    def __init__(self, fpot, ansatz, domain, ipot, backend=Aer.get_backend('aer_simulator'), num_shots=100):

        self.domain = domain
        self.xvect = torch.linspace(domain['xmin'], domain['xmax'],2**ansatz.num_qubits)
        self.dx = 1./(self.xvect[1]-self.xvect[0])
        self.user_potential = fpot
        self.ansatz = ansatz
        self.ipot = ipot
        
        self.backend = backend
        self.NUM_SHOTS = num_shots
        self.params =  np.random.rand(ansatz.num_parameters)

        self.shifted_ansatz = ansatz.compose(ShiftOperator(ansatz.num_qubits))
        self.all_strs = [format(i,'b').zfill(ansatz.num_qubits) for i in range(2**ansatz.num_qubits)]
        self.create_observables()

        self.qi = QuantumInstance(self.backend)
        self.circuit_sampler = CircuitSampler(
            self.backend,
            statevector=is_statevector_backend(self.qi.backend),
            param_qobj=is_aer_provider(self.qi.backend),
        )
        self.initial_point = None
        self.create_potential_observable()
        self.solution = self.get_solution()

        # Initialize the COBYLA optimizer
        self.optimizer = COBYLA(maxiter=100, tol=0.0001, callback=None)

    def create_observables(self):
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        self.observables_dict = {
            "I^(n-1)X" :  TensoredOp((self.ansatz.num_qubits-1) * [I] ) ^ X,
            "Io^(n-1)X":  TensoredOp((self.ansatz.num_qubits-1) * [zero_op] ) ^ X,
            "Io^(n-1)I":  TensoredOp((self.ansatz.num_qubits-1) * [zero_op] ) ^ I,
            "I^(n)O": TensoredOp((self.ansatz.num_qubits) * [I]) ^ one_op,
            "XI^(n)": X ^ TensoredOp((self.ansatz.num_qubits) * [I]) 
        }

        self.observables = [
            self.observables_dict ["I^(n-1)X"],
            self.observables_dict ["I^(n-1)X"] - self.observables_dict ["Io^(n-1)X"]
            ]

    def get_matrix(self):
        """Returns the finite difference matrix
        """

        S = np.real(Operator(ShiftOperator(self.ansatz.num_qubits)).data)

        H1 = np.real(Operator(self.observables_dict ["I^(n-1)X"]).data)
        H2 = S.T @ H1 @ S
        H3 = S.T @ (np.real(Operator(self.observables_dict ["Io^(n-1)X"]).data)) @ S

        size = 2**self.ansatz.num_qubits
        A = -2*np.eye(size) + H1 + H2 - H3
        return A

    def forward(self, x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        raise ValueError("Wave function amplitude not implemented")

    def sort_counts(self,counts):
        return torch.tensor( [counts[k]/self.NUM_SHOTS if k in counts.keys() else 0 for k in self.all_strs]  )


    def sample(self):

        astz = self.ansatz.assign_parameters(self.params, inplace=False)
        astz.measure_all()
        t_qc = transpile(astz, self.backend)
        qobj = assemble(t_qc, shots=self.NUM_SHOTS)
        counts = self.backend.run(qobj).result().get_counts(astz)
        return self.sort_counts(counts)

    def pdf(self, sorted_counts):
        '''density of the wave function.'''
        return torch.tensor(sorted_counts).reshape(-1)

    def nuclear_potential_count(self, sorted_counts):
        return torch.sum(sorted_counts * self.user_potential(self.xvect))

    def create_potential_observable(self):

        map_op = {'0':(I + Z) / 2,'1':(I - Z) / 2}
        self.pot_obs, self.pot_coeff = [], []
        for i, bstr in enumerate(self.all_strs):
            op = map_op[bstr[0]]
            for b in bstr[1:]:
                op = op ^ map_op[b]
            self.pot_obs.append(op)
            self.pot_coeff.append(self.user_potential(self.xvect[i]))

    def nuclear_potential(self):

        circuits = [self.ansatz]*len(self.pot_obs)

        ansatz_params = self.params
        num_parameters = self.ansatz.num_parameters
        parameter_sets = np.reshape(ansatz_params, (-1, num_parameters))
        param_bindings = dict(
            zip(ansatz_params, parameter_sets.transpose().tolist())
        )

        expect_ops = []
        for circ, obs in zip(circuits, self.pot_obs):
            expect_ops.append(self.construct_expectation(ansatz_params, circ, obs))

        out = 0.0
        for c, op in zip(self.pot_coeff, expect_ops):
            sampled_expect_op = self.circuit_sampler.convert(
                op, params=param_bindings
            )
            out += c*np.real(sampled_expect_op.eval()[0])

        return out

    def kinetic_energy_count(self, count):

        wf_val = count.sqrt()

        gr = np.gradient(wf_val, self.dx)
        hs = np.gradient(gr,self.dx)
        k = -0.5*count*torch.tensor(hs)/wf_val
        k[count==0] = 0
        return torch.sum(k)


    @staticmethod
    def construct_expectation(parameter, circuit, observable):
        # assign param to circuit
        wave_function = circuit.assign_parameters(parameter)

        # compose the statefn of the observable on the circuit
        return ~StateFn(observable) @ StateFn(wave_function)

    def kinetic_energy(self):

        circuits = [ self.ansatz, self.shifted_ansatz]

        ansatz_params = self.params
        num_parameters = self.ansatz.num_parameters
        parameter_sets = np.reshape(ansatz_params, (-1, num_parameters))
        param_bindings = dict(
            zip(ansatz_params, parameter_sets.transpose().tolist())
        )
        expect_ops = []
        for circ, obs in zip(circuits, self.observables):
            expect_ops.append(self.construct_expectation(ansatz_params, circ, obs))

        out = -2.0
        for op in expect_ops:
            sampled_expect_op = self.circuit_sampler.convert(
                op, params=param_bindings
            )
            out += np.real(sampled_expect_op.eval()[0])

        return -0.5 * out * (self.dx**2)

    def run(self):

        def callback_opt(res):
            print(res)
            # wfcpy = deepcopy(self)
            # wfcpy.params = res
            # counts = wfcpy.sample().detach().numpy().tolist()
            # points = list(zip(wfcpy.xvect.numpy().tolist(), counts))
            # return points, wfcpy.get_score()            

        def objective_function(params):

            self.params = params
            counts = self.sample()
            v = self.nuclear_potential_count(counts)
            # k = self.kinetic_energy_count(counts)
            # v = counts * self.user_potential(self.xvect)
            k = self.kinetic_energy()
            
            e = k+v
            print('kinetic: ', k, ' potential: ', v, ' total: ', e)            
            return e
            
        # Create the initial parameters (noting that our single qubit variational form has 3 parameters)
        if self.params is None:
            self.params = np.random.rand(self.ansatz.num_parameters)
        ret = self.optimizer.minimize(fun=objective_function, x0=self.params)

        return ret

    def get_solution(self):

        """Computes the solution using finite difference
        
        Args:
            npts (int, optional): number of discrete points. Defaults to 100.

        Returns:
            dict: position and numerical value of the wave function 
        """
        # npts = len(self.xvect)
        # x = self.xvect
        # dx2 = self.dx**2
        # Vx = np.diag(self.user_potential(self.xvect).detach().numpy().flatten())
        # K = -0.5 / dx2.numpy() * ( np.eye(npts,k=1) + np.eye(npts,k=-1) - 2. * np.eye(npts))
        # l, U = np.linalg.eigh(K+Vx)
        # sol = np.abs(U[:,0])
        

        x = self.xvect

        if self.ipot == 0:
            sol = np.exp(-0.5*x**2).numpy().astype('float64')
        elif self.ipot == 1:
            x=x+2
            sol = np.exp(-np.exp(-x)-0.5*x).numpy().astype('float64')
        elif self.ipot == 2:
            sol = (np.exp(-np.abs(x-2.5)) + np.exp(-np.abs(x+2.5))).numpy().astype('float64')

        return {'x':x,'y':sol,'max':np.max(sol)}
        

    def get_score(self):
        
        with torch.no_grad():

            ywf = self.sample().clone().detach().numpy()
            ywf = np.sqrt(ywf)
            ywf = (ywf/np.max(ywf) * self.solution['max']).flatten()
            return self._score(ywf)

    def _score(self,yvals):
    
        d = np.sqrt(np.sum((self.solution['y']-yvals)**2))
        return np.exp(-d)