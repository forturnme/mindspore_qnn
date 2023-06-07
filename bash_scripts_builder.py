"""
build arbitary bash run scripts for qnn.
"""
import os

# if logs dir is not exist, create it
if not os.path.exists('logs'):
    os.makedirs('logs')

# define valid qnn styles and datasets.
style = ['u3cu3', 'cnot_zxz', 'rxyz', 'zz_ry', 'zx_xx']
dataset = ['mnist4x4', 'fashion4x4', 'vowel']

# parse args.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qubits", type=int)
parser.add_argument("--dataset", type=str, default='mnist4x4')
args = parser.parse_args()
print('creating test bash scripts for qnns...')
print('chosen qubits:', args.qubits)
print('chosen dataset:', args.dataset)

# check if the args are valid.
if args.dataset not in dataset:
    raise ValueError('the dataset is not valid.')
if args.qubits<10 and dataset in ['mnist4x4', 'fashion4x4']:
    raise ValueError('the number of qubits should be larger than classes of a dataset.')
if args.qubits<11 and dataset=='vowel':
    raise ValueError('the number of qubits should be larger than classes of a dataset.')

# warn user if the number of qubits is larger than 12.
if args.qubits>12:
    print(f'warning: the number of qubits ({args.qubits}) is larger than 12, this may cause memory error.')

# build bash scripts.
# for mnist4x4 and fashion4x4, we use net/mnist4x4_*.py.
# for vowel, we use net/vowel_*.py.
print('building bash scripts...')
bash_script = ""
if args.dataset in ['mnist4x4', 'fashion4x4']:
    # test from 2 to 10 classes.
    # test configurations for qnnn and qnnl (layers, reuploads):
    #   (4, 4), (8, 4), (1, 16), (1, 32)
    # test configurations for linear: no need to specify layers and uploads, just set them as 1.
    # also, qubits for linear is 0.
    # test configurations for ann: layers and uploads is 1, but n_hidden in [2, 10, 100] instead of n_qubits.
    # for ann and linear, style is ann or linear.
    # for qnnn and qnnl, style is one of the five styles, and n_qubits is [10, 12].
    # path and name for output files should be like: 
    #       logs/{dataset}/[linear,ann,qnnn,qnnl]_{style}_q{n_qubits}_l{n_layers}_r{n_reuploads}_c{n_classes}.log
    # first, add the scripts for linear and ann, and run them simultaneously.
    # just dont care about the stderr stream.
    # first run anns.
    for n_hidden in [2, 10, 100]:
        for n_classes in range(2,11):
            bash_script += f'nohup python net/mnist4x4_ann.py --classes {n_classes} --qubits {n_hidden} --dataset {args.dataset} --reuploads 1 --layers 1 --style ann > logs/{args.dataset}/ann_ann_q{n_hidden}_l1_r1_c{n_classes}.log &\n'
        # after starting run anns, sleep for 8 minutes.
        bash_script += 'sleep 480\n'
    # then run linear.
    for n_classes in range(2,11):
        bash_script += f'nohup python net/mnist4x4_linear.py --classes {n_classes} --qubits 0 --dataset {args.dataset} --reuploads 1 --layers 1 --style linear > logs/{args.dataset}/linear_linear_q0_l1_r1_c{n_classes}.log &\n'
    # after starting run linear, sleep for 5 minutes.
    bash_script += 'sleep 300\n'
    # and then run qnnn at qubits. 10 and 12 is recommended.
    for netname in ['qnnn', 'qnnl']:
        for style in ['u3cu3', 'cnot_zxz', 'rxyz', 'zz_ry', 'zx_xx']:
            for qubits in [args.qubits]:
                for layers, uploads in [(4,4), (8,4), (1,16), (1,32)]:
                    for n_classes in range(2,11):
                        bash_script += f'nohup python net/mnist4x4_{netname}.py --classes {n_classes} --qubits {qubits} --dataset {args.dataset} --reuploads {uploads} --layers {layers} --style {style} > logs/{args.dataset}/{netname}_{style}_q{qubits}_l{layers}_r{uploads}_c{n_classes}.log &\n'
                    # after starting run qnnn, sleep for 1 hour.
                    bash_script += 'sleep 3600\n'

elif args.dataset == 'vowel':
    # test from 2 to 11 classes.
    # test configurations for qnnn and qnnl (layers, reuploads):
    #   (4, 4), (8, 4), (1, 16), (1, 32)
    # test configurations for linear: no need to specify layers and uploads, just set them as 1.
    # also, qubits for linear is 0.
    # test configurations for ann: layers and uploads is 1, but n_hidden in [2, 10, 100] instead of n_qubits.
    # for ann and linear, style is ann or linear.
    # for qnnn and qnnl, style is one of the five styles, and n_qubits is [11, 12].
    # path and name for output files should be like: 
    #       logs/{dataset}/[linear,ann,qnnn,qnnl]_{style}_q{n_qubits}_l{n_layers}_r{n_reuploads}_c{n_classes}.log
    # first, add the scripts for linear and ann, and run them simultaneously.
    # just dont care about the stderr stream.
    # first run anns.
    for n_hidden in [2, 10, 100]:
        for n_classes in range(2,12):
            bash_script += f'nohup python net/vowel_ann.py --classes {n_classes} --qubits {n_hidden} --dataset {args.dataset} --reuploads 1 --layers 1 --style ann > logs/{args.dataset}/ann_ann_q{n_hidden}_l1_r1_c{n_classes}.log &\n'
        # after starting run anns, sleep for 5 mins.
        bash_script += 'sleep 300\n'
    # then run linear.
    for n_classes in range(2,12):
        bash_script += f'nohup python net/vowel_linear.py --classes {n_classes} --qubits 0 --dataset {args.dataset} --reuploads 1 --layers 1 --style linear > logs/{args.dataset}/linear_linear_q0_l1_r1_c{n_classes}.log &\n'
    # after starting run linear, sleep for 5 mins.
    bash_script += 'sleep 300\n'
    # and then run qnnn at qubits. 11 and 12 is recoommended.
    for netname in ['qnnn', 'qnnl']:
        for style in ['u3cu3', 'cnot_zxz', 'rxyz', 'zz_ry', 'zx_xx']:
            for qubits in [args.qubits]:
                for layers, uploads in [(4,4), (8,4), (1,16), (1,32)]:
                    for n_classes in range(2,12):
                        bash_script += f'nohup python net/vowel_{netname}.py --classes {n_classes} --qubits {qubits} --dataset {args.dataset} --reuploads {uploads} --layers {layers} --style {style} > logs/{args.dataset}/{netname}_{style}_q{qubits}_l{layers}_r{uploads}_c{n_classes}.log &\n'
                    # after starting run qnnn, sleep for 10 mins.
                    bash_script += 'sleep 600\n'

# write bash scripts to file.
with open(f'run_tests_for_{args.dataset}_q{args.qubits}.sh', 'w') as f:
    f.write(bash_script)
                
print(f'bash scripts are saved to ./run_tests_for_{args.dataset}_q{args.qubits}.sh.')
print('done.')

