'''
build qnn circuits with mindquantum
including qnn circuits and encoder circuits
'''

# import libraries
import numpy as np

# import mindquantum libraries
import mindquantum as mq

def qnn_zz_ry(layers, n_qubits, ring = False, prefix='qnn'):
    '''
    return a qnn circuit build over mindquantum
    with no input layer but just the qnn layers
    having layers number of layers
    and n_qubits number of qubits
    the circuit block style is zz-ry
    if ring is True, then the zz layer is in a ring form
    otherwise, the zz layer is in a linear form  
    linear form: q0-q1-q2-q3
    ring form: q0-q1-q2-q3-q0
    all ry gates are parameterized with separate parameters, starting with prefix 'ry'
        and ending with the layer number and the qubit number, with 6 digits each.
            e.g., ry_000001_000001
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # add the qnn layers
    for i in range(layers):
        # add the zz layer
        for j in range(n_qubits-1):
            circuit += mq.ZZ(prefix+'zz_{:06d}_{:06d}'.format(j, j+1)).on([j, j+1])
        if ring:
            circuit += mq.ZZ(prefix+'zz_{:06d}_{:06d}'.format(n_qubits-1, 0)).on([n_qubits-1, 0])
        # add the ry layer
        for j in range(n_qubits):
            circuit += mq.RY(prefix+'ry_{:06d}_{:06d}'.format(i, j)).on(j)
    # return the circuit
    return circuit


def qnn_u3_cu3(layers, n_qubits, ring = True, prefix='qnn'):
    '''
    return a qnn circuit build over mindquantum
    with no input layer but just the qnn layers
    having layers number of layers
    and n_qubits number of qubits
    the circuit block style is u3-cu3
    if ring is True, then the cu3 layer is in a ring form
    otherwise, the cu3 layer is in a linear form  
    linear form: q0-q1-q2-q3
    ring form: q0-q1-q2-q3-q0
    all u3 and cu3 gates are parameterized with separate 3 parameters each, starting with prefix of 'u31', 'u32', 'u33', 'cu31', 'cu32', 'cu33'
        and ending with the layer number and the qubit number, with 6 digits each.
            e.g., u3_000001_000001, cu3_000001_000001
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # add the qnn layers
    for i in range(layers):
        # add the u3 layer
        for j in range(n_qubits):
            circuit += mq.U3(prefix+'u31_{:06d}_{:06d}'.format(i, j), 
                             prefix+'u32_{:06d}_{:06d}'.format(i, j),
                             prefix+'u33_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the cu3 layer
        for j in range(n_qubits-1):
            circuit += mq.U3(prefix+'cu31_{:06d}_{:06d}'.format(i, j), 
                             prefix+'cu32_{:06d}_{:06d}'.format(i, j), 
                             prefix+'cu33_{:06d}_{:06d}'.format(i, j)).on(j, j+1)
        if ring:
            circuit += mq.U3(prefix+'cu31_{:06d}_{:06d}'.format(i, n_qubits-1), 
                             prefix+'cu32_{:06d}_{:06d}'.format(i, n_qubits-1), 
                             prefix+'cu33_{:06d}_{:06d}'.format(i, n_qubits-1)).on(n_qubits-1, 0)
    # return the circuit
    return circuit


def qnn_zx_xx(layers, n_qubits, ring = False, prefix='qnn'):
    '''
    return a qnn circuit build over mindquantum
    with no input layer but just the qnn layers
    having layers number of layers
    and n_qubits number of qubits
    the circuit block style is rz-rx-xx
    xx stands for mq.Rxx gates
    if ring is True, then the zz layer is in a ring form
    otherwise, the xx layer is in a linear form  
    linear form: q0-q1-q2-q3
    ring form: q0-q1-q2-q3-q0
    all z and x and xx gates are parameterized with separate parameters, starting with prefix of their names
        and ending with the layer number and the qubit number, with 6 digits each.
            e.g., z_000001_000001, x_000001_000001, xx_000001_000001
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # add the qnn layers
    for i in range(layers):
        # add the z layer
        for j in range(n_qubits):
            circuit += mq.RZ(prefix+'z_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the x layer
        for j in range(n_qubits):
            circuit += mq.RX(prefix+'x_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the xx layer
        for j in range(n_qubits-1):
            circuit += mq.XX(prefix+'xx_{:06d}_{:06d}'.format(i, j)).on((j, j+1))
        if ring:
            circuit += mq.XX(prefix+'xx_{:06d}_{:06d}'.format(i, n_qubits-1)).on((n_qubits-1, 0))
    # return the circuit
    return circuit


def qnn_cnot_zxz(layers, n_qubits, ring = False, prefix='qnn'):
    '''
    return a qnn circuit build over mindquantum
    with no input layer but just the qnn layers
    having layers number of layers
    and n_qubits number of qubits
    the circuit block style is cnot-rz-rx-rz
    if ring is True, then the zz layer is in a ring form
    otherwise, the zz layer is in a linear form  
    linear form: q0-q1-q2-q3
    ring form: q0-q1-q2-q3-q0
    all z and x gates are parameterized with separate parameters, starting with prefix 'z1', 'z2' or 'x'
        and ending with the layer number and the qubit number, with 6 digits each.
            e.g., z1_000001_000001, x_000001_000001, z2_000001_000001
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # add the qnn layers
    for i in range(layers):
        # add the cnot layer
        for j in range(n_qubits-1):
            circuit += mq.CNOT.on(j, j+1)
        if ring:
            circuit += mq.CNOT.on(n_qubits-1, 0)
        # add the z layer
        for j in range(n_qubits):
            circuit += mq.RZ(prefix+'z1_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the x layer
        for j in range(n_qubits):
            circuit += mq.RX(prefix+'x_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the z layer
        for j in range(n_qubits):
            circuit += mq.RZ(prefix+'z2_{:06d}_{:06d}'.format(i, j)).on(j)
    # return the circuit
    return circuit


def qnn_rxyz_swap(layers, n_qubits, prefix='qnn'):
    '''
    return a qnn circuit build over mindquantum
    with no input layer but just the qnn layers
    having layers number of layers
    and n_qubits number of qubits
    the circuit block style is rz-ry-rx-swap
    there is only one swap connection style:
        nearest neighbor swap: e.g., the first part of a swap layer is q0-q1, q2-q3, q4-q5, ..., (qn-1)-qn
            and the second part is q1-q2, q3-q4, q5-q6, ..., (qn-2)-(qn-1)
    swap is realized by mq.RZ gates
    all z and x gates are parameterized with separate parameters, starting with prefix 'z', 'y' or 'x'
        and ending with the layer number and the qubit number, with 6 digits each.
            e.g., z_000001_000001, x_000001_000001, y_000001_000001
    whatelse, there is one fixed mq.RZ(np.pi/4) layer before all the circuit
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # add the fixed rz layer
    for j in range(n_qubits):
        circuit += mq.RZ(np.pi/4).on(j)
    # add the qnn layers
    for i in range(layers):
        # add the rz layer
        for j in range(n_qubits):
            circuit += mq.RZ(prefix+'z_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the ry layer
        for j in range(n_qubits):
            circuit += mq.RY(prefix+'y_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the rx layer
        for j in range(n_qubits):
            circuit += mq.RX(prefix+'x_{:06d}_{:06d}'.format(i, j)).on(j)
        # add the swap layer
        for j in range(n_qubits//2):
            circuit += mq.Z.on(2*j, 2*j+1)
        for j in range(n_qubits//2-1):
            circuit += mq.Z.on(2*j+1, 2*j+2)
    # return the circuit
    return circuit


# now define the encoder circuits
# the encoder circuit is a circuit that encodes the input data into the qubits
# has 3 types: xyz encoder and iqp encoder and amplitude encoder

def xyz_encoder(n_dim, n_qubits):
    '''
    generate circuits that encode the n_dim input vectors to parameters of rx, ry and rz gates
    the circuit is a linear circuit
    the circuit has n_qubits qubits
    vectors are encoded layerwise, e.g., the first rx layer encodes the first n_qubits input elements,
    then the following ry layer encodes the [n_qubits:2*n_qubits] input elements, and so on.
    the layer sequence is rx-ry-rz-rx-ry-rz-...
    until a total of n_dim of rx, ry and rz layers are added
    the parameter of the input gates are named after the prefix 'enc' and the n-th element of the input vector,
        with 6 digits each.
            e.g., enc_000001, enc_000002, enc_000003, ...
    '''
    # initialize the circuit
    circuit = mq.Circuit()
    # calculate the number of layers
    n_layers = n_dim//(n_qubits*3)
    n_reminder = n_dim%(n_qubits*3)
    # add the encoder layers
    for i in range(n_layers):
        # add the rx layer
        for j in range(n_qubits):
            circuit += mq.RX('enc_{:06d}'.format(3*i*n_qubits+j)).on(j)
        # add the ry layer
        for j in range(n_qubits):
            circuit += mq.RY('enc_{:06d}'.format(3*i*n_qubits+n_qubits+j)).on(j)
        # add the rz layer
        for j in range(n_qubits):
            circuit += mq.RZ('enc_{:06d}'.format(3*i*n_qubits+2*n_qubits+j)).on(j)
    # add the reminder rx layer
    added = 0
    for j in range(n_qubits):
        if added >= n_reminder:
            break
        circuit += mq.RX('enc_{:06d}'.format(3*n_layers*n_qubits+j)).on(j)
        added += 1
    # add the reminder ry layer
    for j in range(n_qubits):
        if added >= n_reminder:
            break
        circuit += mq.RY('enc_{:06d}'.format(3*n_layers*n_qubits+n_qubits+j)).on(j)
        added += 1
    # add the reminder rz layer
    for j in range(n_qubits):
        if added >= n_reminder:
            break
        circuit += mq.RZ('enc_{:06d}'.format(3*n_layers*n_qubits+2*n_qubits+j)).on(j)
        added += 1
    # return the circuit
    return circuit


def iqp_encoder(n_dim, n_qubits):
    '''
    iqp encoders: mq.algorithm.nisq.IQPEncoding
    generate circuits that encode the n_dim input vectors
    need modifications. TODO
    '''
    # raise NotImplementedError
    # initialize the circuit
    idx = 0
    circuit = mq.UN(mq.H, n_qubits) # add the hadamard layer
    while idx < n_dim:
        sidx = idx
        # add the mq.RZ layer
        for qb in range(n_qubits):
            circuit += mq.RZ('enc_{:06d}'.format(idx)).on(qb)
            idx += 1
            if idx >= n_dim:
                break
        # add the qp layer
        for i in range(1, n_qubits):
            circuit += mq.X.on(i, i - 1)
            circuit += mq.RZ(f'enc_{sidx+i-1:06d} * enc_{sidx+i:06d}').on(i)
            circuit += mq.X.on(i, i - 1)
        circuit += mq.BARRIER
        circuit = mq.UN(mq.H, n_qubits) # add the hadamard layer
        if sidx+i >= n_dim:
            break
    # return the circuit
    return circuit


def iqp_data_prepare(inputs, n_qubits):
    '''
    iqp data prepare
    prepare the input data for iqp encoder
    add 'enc_{sidx+i-1:06d} * enc_{sidx+i:06d}' after every n_qubits input elements
    output the prepared input vector as np.array
    '''
    # initialize the circuit
    idx = 0
    output = []
    # one input element layer has up to n_qubits elements
    layers = len(inputs) // n_qubits
    has_remainder = len(inputs) % n_qubits
    li = 0
    while li < layers:
        sidx = idx
        for i in range(n_qubits):
            output.append(inputs[idx])
            idx += 1
        for i in range(1, n_qubits):
            output.append(inputs[sidx-1+i] * inputs[sidx+i])
        li += 1
    if has_remainder:
        sidx = idx
        for i in range(has_remainder):
            output.append(inputs[idx])
            idx += 1
        for i in range(1, has_remainder):
            output.append(inputs[sidx-1+i] * inputs[sidx+i])
    # return the output
    return np.array(output)
    

def amplitude_encoder(inputs, n_qubits):
    '''
    amplitude encoder realized by the amplitude embedding circuit.
    directly use the mindquantum.algorithm.library.amplitude_encoder
    return circuit and parameters for the encoder.
    '''
    return mq.algorithm.library.amplitude_encoder(inputs, n_qubits)


def hams_for_classification(n_class):
    '''
    set hams list for classification.
    '''
    # initialize the hams list
    hams = []
    # add the hamiltonians
    for i in range(n_class):
        hams.append(mq.Hamiltonian(mq.QubitOperator('X' + str(i), 1.0)))
    return hams

