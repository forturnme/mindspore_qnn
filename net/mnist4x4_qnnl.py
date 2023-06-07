'''
data reuploading quantum neural network.
'''
import sys
import numpy as np                                        # 导入numpy库并简写为np
# import os
# os.environ["OMP_NUM_THREADS"] = '8'

# pylint: disable=W0104
from mindquantum.core.circuit import Circuit         # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.circuit import UN              # 导入UN模块
from mindquantum.core.gates import H, X, RZ          # 导入量子门H, X, RZ

from qnn_circuits import xyz_encoder, qnn_u3_cu3, hams_for_classification, qnn_cnot_zxz, qnn_rxyz_swap, qnn_zz_ry, qnn_zx_xx
# from qnn_circuits_id import qnn_id_xyz_u3cu3

# from download import download

# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#       "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--classes", type=int)
parser.add_argument("--layers", type=int)
parser.add_argument("--reuploads", type=int)
parser.add_argument("--qubits", type=int)
parser.add_argument("--dataset", type=str, default='mnist4x4')
parser.add_argument("--style", type=str, default='u3cu3')
args = parser.parse_args()
print('qnn followed by a linear classifier.')
print('chosen classes:', args.classes)
print('chosen layers:', args.layers)
print('chosen qubits:', args.qubits)
print('chosen reuploads:', args.reuploads)
print('chosen dataset:', args.dataset)
print('chosen style:', args.style)


n_dim = 4*4
n_qubits = args.qubits
batch_size = 256
lr = 0.005
# lr = 0.01
weight_decay = 0.0001
epochs = 150

if args.dataset == 'mnist4x4':
    datadir = 'data/mnist4x4'
elif args.dataset == 'fashion4x4':
    datadir = 'data/fashion4x4'

if args.style == 'u3cu3':
    qnn_module = qnn_u3_cu3
elif args.style == 'cnot_zxz':
    qnn_module = qnn_cnot_zxz
elif args.style == 'rxyz':
    qnn_module = qnn_rxyz_swap
elif args.style == 'zz_ry':
    qnn_module = qnn_zz_ry
elif args.style == 'zx_xx':
    qnn_module = qnn_zx_xx


# build encoder circuit
encoder = xyz_encoder(n_dim, n_qubits)
encoder = encoder.no_grad()     
# encoder.summary()

# build qnn circuit
reuploads = args.reuploads
n_layers = args.layers
# n_classes = 4
# n_layers = 96
qnn_reups = []
for i in range(reuploads):
    qnn_reups.append(qnn_module(n_layers, n_qubits, ring=True, prefix=str(i)))
# qnn.summary()

# build hamiltonians
hams = hams_for_classification(n_qubits)

# build dru circuit
circuit = Circuit()
for reup in qnn_reups:
    circuit += encoder.as_encoder() + reup.as_ansatz()
# circuit = qnn_id_xyz_u3cu3(n_layers, n_qubits, n_dim, reuploads)

# build mindspore model
# pylint: disable=W0104
import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer                                                      # 导入MQLayer
from mindquantum.simulator import Simulator

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(42)                                                                                 # 设置生成随机数的种子
np.random.seed(42)
sim = Simulator('mqvector', n_qubits)

# test duplicating ansatz
# circuit = circuit + circuit + circuit

circuit.summary()                                                    # 打印量子线路的信息

grad_ops = sim.get_expectation_with_grad(hams,
                                         circuit,
                                         parallel_worker=256)
QuantumNet = MQLayer(grad_ops, weight='normal')          # 搭建量子神经网络

# prepare mnist data
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset, NumpySlicesDataset, CSVDataset
def datapipe(path, batch_size):
    # image_transforms = [
    #     vision.Rescale(1.0 / 255.0, 0),
    #     vision.Normalize(mean=(0.1307,), std=(0.3081,)),
    #     # vision.Rescale(3.14, 0),
    #     vision.CenterCrop(24),
    #     vision.Resize((4, 4)),
    #     vision.HWC2CHW(),
    #     lambda img: img.flatten()
    # ]
    label_transform = transforms.TypeCast(mindspore.int32)
    image_transform = transforms.TypeCast(mindspore.float32)

    # dataset = MnistDataset(path, num_samples=3000)
    data = {'image':np.loadtxt(path[1], delimiter=','),
            'label':np.loadtxt(path[0], delimiter=',')}
    dataset = NumpySlicesDataset(data, num_samples=args.classes*300)
    # for i in range(10):
    #     print(dataset[i])
    # sys.exit(0)
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.map(image_transform, 'image')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe([f'{datadir}/{args.dataset}_{args.classes}_train_labels.csv',
                            f'{datadir}/{args.dataset}_{args.classes}_train_images.csv'], batch_size)
test_dataset = datapipe([f'{datadir}/{args.dataset}_{args.classes}_test_labels.csv',
                            f'{datadir}/{args.dataset}_{args.classes}_test_images.csv'], batch_size)

# train the model
from mindspore.nn import CrossEntropyLoss                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam, Accuracy                                                  # 导入Adam模块用于定义优化参数
from mindspore.train import Model, LossMonitor                       # 导入Accuracy模块，用于评估预测准确率
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset                               # 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集

from time import time

loss = CrossEntropyLoss()            # 通过SoftmaxCrossEntropyWithLogits定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值                

# model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})             # 建立模型：将MindSpore Quantum构建的量子机器学习层和MindSpore的算子组合，构成一张更大的机器学习网络

class Network(nn.Cell):
    def __init__(self, quantumnet, classes=10):
        super().__init__()
        self.quantumnet = quantumnet
        self.classes = classes
        self.linear = nn.Dense(args.qubits, classes, has_bias=True)
        # self.reshape = ops.Reshape()
        # self.activation = nn.Sigmoid()
        self.activation = nn.LeakyReLU()
        # self.classifier = nn.Softmax()
        # self.reshape2 = ops.Reshape()

    def construct(self, x):
        # x = self.flatten(x)
        logits = self.quantumnet(x)
        # logits = self.reshape(logits, (len(x), 1, self.classes))
        logits = self.linear(logits)
        logits = self.activation(logits)
        # logits = self.classifier(logits)
        # print(logits.shape)
        # logits = self.reshape2(logits, (len(x), self.classes))
        return logits

model = Network(QuantumNet,classes=args.classes)

opti = Adam(model.trainable_params(), learning_rate=lr, weight_decay=weight_decay)  

def train_loop(model, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        # print('step 1')
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        elapsed2 = time()
        loss = train_step(data, label)

        if batch % 10 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}], \
                  batch time: {time()-elapsed2:>3f}s, estimated epoch time: {(size)*(time()-elapsed2):>3f}s")
        
        sys.stdout.flush()


def test_loop(model, dataset, loss_fn):
    elapsed = time()
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print('test time: ', time()-elapsed, 's')

    sys.stdout.flush()


loss_fn = nn.CrossEntropyLoss()

# ms.save_checkpoint(model, 'checkpoint_test.ckpt')
# ms.load_checkpoint('checkpoint_test.ckpt', model)
epochs = epochs
for t in range(epochs):
    stime = time()
    print(f"Epoch {t+1}")
    train_loop(model, train_dataset, loss_fn, opti)
    print('training 1 epoch time: ', time()-stime)
    print('-'*42)
    if  (t+1)%5==0:
        test_loop(model, test_dataset, loss_fn)
print("Done!")

