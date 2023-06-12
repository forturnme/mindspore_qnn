'''
provide some useful functions.
'''
import numpy as np
import matplotlib.pyplot as plt
import os


# define all different markers for plotting
# https://matplotlib.org/3.1.1/api/markers_api.html
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']


def read_log(logfile):
    '''
    read log file and return the run information.
    the run information is a list of tuple (dataset_name, information)
    the `information` is a dict with keys:
        'dataset': the dataset name
        'style': the network style in 'ann', 'linear', 'qnnn' and 'qnnl'
        'type': the network type in 'qnnn', 'qnnl' and 'ann', 'linear'
        'n_qubits': the number of qubits
        'n_layers': the number of layers
        'n_reuploads': the number of reuploads
        'n_classes': the number of classes
        'losses': a list of loss values
        'epochs': a list of epoch values
        'test_accruacies': a list of accuracy values
        'test_losses': a list of test loss values
    '''
    with open(logfile, 'r') as f:
        lines = f.readlines()
    flap = False
    losses = []
    epochs = []
    test_accruacies = []
    test_losses = []
    for i, line in enumerate(lines):
        if line.startswith('chosen '):
            if line[7:].startswith('dataset: '):
                dataset = line[15:].strip()
            elif line[7:].startswith('classes: '):
                n_classes = int(line[16:])
            elif line[7:].startswith('layers: '):
                n_layers = int(line[15:])
            elif line[7:].startswith('qubits: '):
                n_qubits = int(line[15:])
            elif line[7:].startswith('reuploads: '):
                n_reuploads = int(line[17:])
            elif line[7:].startswith('style: '):
                style = line[14:].strip()
        elif line.startswith('Pure Linear classifier.'):
            type = 'linear'
            n_layers = 1
            n_reuploads = 1
            n_qubits = 0
        elif line.startswith('qnn followed by a linear classifier.'):
            type = 'qnnl'
        elif line.startswith('MLP with n_qubits hidden neurons.'):
            type = 'ann'
            n_layers = 1
            n_reuploads = 1
        elif line.startswith('naive qnn not followed by a linear classifier.'):
            type = 'qnnn'
        else:
            if line.startswith('Epoch '):
                flap = True
                epochs.append(int(line[6:]))
            if flap and line.startswith('loss: '):
                losses.append(float(line[6:13]))
                flap = False
            if line.startswith(' Accuracy: '):
                test_accruacies.append(float(line[10:line.find('%')]))
                test_losses.append(float(line[line.find('loss: ') + 6:]))
    return {
        'type': type,
        'dataset': dataset,
        'style': style,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_reuploads': n_reuploads,
        'n_classes': n_classes,
        'losses': losses,
        'epochs': epochs,
        'test_accruacies': test_accruacies,
        'test_losses': test_losses
    }


def read_log_dir(logdir):
    """
    read all .log files under logdir.
    return a list of run information.
    """
    logs = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.endswith('.log'):
                logs.append(read_log(os.path.join(root, file)))
    return logs    


def plot_accruacy_netconf(logs, style=None, dataset=None, qubits=None, reuploads=None,
                          layers=None, type=None, save_path='figs/', figsize=(10, 8)):
    """
    plot accruacy to n_classes plots for different network configurations.
    accruacies is measured every 5 epochs.
    for each network configuration, plot the best accruacy for every classes.
    """
    # filter logs
    if style is not None:
        logs = [log for log in logs if log['style'] == style]
    if type is not None:
        logs = [log for log in logs if log['type'] == type]
    if dataset is not None:
        logs = [log for log in logs if log['dataset'] == dataset]
    if qubits is not None:
        logs = [log for log in logs if log['n_qubits'] == qubits]
    if reuploads is not None:
        logs = [log for log in logs if log['n_reuploads'] == reuploads]
    if layers is not None:
        logs = [log for log in logs if log['n_layers'] == layers]
    # make plot data
    plot_data = {}
    for log in logs:
        netname = ''
        if style is not None:
            pass
        else:
            netname += log['style'] + '_'
        if type is not None:
            pass
        else:
            netname += log['type'] + '_'
        if dataset is not None:
            pass
        else:
            netname += log['dataset'] + '_'
        if qubits is not None:
            pass
        else:
            netname += str(log['n_qubits']) + 'q_'
        if reuploads is not None:
            pass
        else:
            netname += str(log['n_reuploads']) + 'r_'
        if layers is not None:
            pass
        else:
            netname += str(log['n_layers']) + 'l'
        if netname not in plot_data:
            plot_data[netname] = {
                'n_classes': [],
                'accruacies': []
            }
        plot_data[netname]['n_classes'].append(log['n_classes'])
        plot_data[netname]['accruacies'].append(max(log['test_accruacies']))
    # sort n_classes, and sort accruacies according to n_classes
    for netname in plot_data:
        plot_data[netname]['n_classes'], plot_data[netname]['accruacies'] = \
            zip(*sorted(zip(plot_data[netname]['n_classes'], plot_data[netname]['accruacies'])))
    # plot
    plt.figure(figsize=figsize)
    # make title
    title = 'Accuracy to number of classes: '
    filename = 'ACC_NCLS_'
    if style is not None:
        title += style + ' '
        filename += style + '_'
    if type is not None:
        title += type + ' '
        filename += type + '_'
    if dataset is not None:
        title += dataset + ' '
        filename += dataset + '_'
    if qubits is not None:
        title += str(qubits) + 'qubits '
        filename += str(qubits) + 'q_'
    if reuploads is not None:
        title += str(reuploads) + 'reuploads '
        filename += str(reuploads) + 'r_'
    if layers is not None:
        title += str(layers) + 'layers'
        filename += str(layers) + 'l'
    plt.title(title)
    plt.xlabel('Number of classes')
    plt.ylabel('Accuracy')
    # plot
    for i, netname in enumerate(plot_data):
        plt.plot(plot_data[netname]['n_classes'], plot_data[netname]['accruacies'],
                 marker=markers[i%len(markers)], label=netname)
    plt.legend()
    plt.savefig(save_path + filename + '.pdf')
    plt.show()


def plot_accruacy_epoch(logs, style=None, dataset=None, qubits=None, reuploads=None,
                        layers=None, type=None, save_path='figs/', figsize=(10, 8)):
    """
    plot accruacy to epoch plots for different network configurations.
    every n_classes is for a label.
    must have all parameters to proceed, else raise error.
    """
    # check if all parameters are given
    if style is None or dataset is None or qubits is None or reuploads is None or layers is None\
            or type is None:
        raise ValueError('must have all parameters to proceed.')
    # filter logs
    logs = [log for log in logs if log['style'] == style]
    logs = [log for log in logs if log['type'] == type]
    logs = [log for log in logs if log['dataset'] == dataset]
    logs = [log for log in logs if log['n_qubits'] == qubits]
    logs = [log for log in logs if log['n_reuploads'] == reuploads]
    logs = [log for log in logs if log['n_layers'] == layers]
    # make plot title and filename
    title = 'Accuracy to epoch: '
    filename = 'ACC_EPOCH_'
    # all parameters are given, so add them to title and filename
    title += style + ' ' + type + ' ' + dataset + ' ' + str(qubits) + 'qubits ' + \
        str(reuploads) + 'reuploads ' + str(layers) + 'layers'
    filename += style + '_' + type + ' ' + dataset + '_' + str(qubits) + 'q_' + \
        str(reuploads) + 'r_' + str(layers) + 'l'
    # make plot data
    classes = []
    plot_data = {}
    for log in logs:
        series_name = str(log['n_classes']) + ' classes'
        if series_name not in plot_data:
            classes.append(log['n_classes'])
            plot_data[series_name] = {
                'epochs': [],
                'accruacies': []
            }
        plot_data[series_name]['accruacies'] = log['test_accruacies']
        plot_data[series_name]['epochs'] = list(range(1, len(log['test_accruacies']) + 1, 5))
    # plot
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for class_ in sorted(classes):
        plt.plot(plot_data[str(class_) + ' classes']['epochs'],
                 plot_data[str(class_) + ' classes']['accruacies'],
                 label=str(class_) + ' classes')
    plt.legend()
    plt.savefig(save_path + filename + '.pdf')
    plt.show()


def plot_loss_epoch(logs, style=None, dataset=None, qubits=None, reuploads=None,
                    layers=None, type=None, save_path='figs/', figsize=(10, 8)):
    """
    plot loss to epoch plots for different network configurations.
    every n_classes is for a label.
    must have all parameters to proceed, else raise error.
    """
    # check if all parameters are given
    if style is None or dataset is None or qubits is None or reuploads is None or layers is None or\
            type is None:
        raise ValueError('must have all parameters to proceed.')
    # filter logs
    logs = [log for log in logs if log['style'] == style]
    logs = [log for log in logs if log['type'] == type]
    logs = [log for log in logs if log['dataset'] == dataset]
    logs = [log for log in logs if log['n_qubits'] == qubits]
    logs = [log for log in logs if log['n_reuploads'] == reuploads]
    logs = [log for log in logs if log['n_layers'] == layers]
    # make plot title and filename
    title = 'Loss to epoch: '
    filename = 'LOSS_EPOCH_'
    # all parameters are given, so add them to title and filename
    title += style + ' ' + type + ' ' + dataset + ' ' + str(qubits) + 'qubits ' + \
        str(reuploads) + 'reuploads ' + str(layers) + 'layers'
    filename += style + '_' + type + ' ' + dataset + '_' + str(qubits) + 'q_' + \
        str(reuploads) + 'r_' + str(layers) + 'l'
    # make plot data
    classes = []
    plot_data = {}
    for log in logs:
        series_name = str(log['n_classes']) + ' classes'
        if series_name not in plot_data:
            classes.append(log['n_classes'])
            plot_data[series_name] = {
                'epochs': [],
                'losses': []
            }
        plot_data[series_name]['losses'] = log['losses']
        plot_data[series_name]['epochs'] = log['epochs']
    # plot
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for class_ in sorted(classes):
        plt.plot(plot_data[str(class_) + ' classes']['epochs'],
                 plot_data[str(class_) + ' classes']['losses'],
                 label=str(class_) + ' classes')
    plt.legend()
    plt.savefig(save_path + filename + '.pdf')
    plt.show()


def plot_accruacy_netconf_trebled(logs, style=None, dataset=None, qubits=None,
                                reuploads=None, layers=None, type=None,
                                t_style=None, t_dataset=None, t_qubits=None,
                                t_reuploads=None, t_layers=None, t_type=None,
                                save_path='figs/', figsize=(10, 8)):
    """
    plot accruacy to classes plots for different network configurations,
    and treble (highlight) some of them.
    all trebled nets have an alpha of 1, others have 0.3.
    the t_parameters must be provided at least one.
    """
    # check if one t_parameter is given
    if t_style is None and t_dataset is None and t_qubits is None and t_reuploads is None and\
          t_layers is None and t_type is None:
        raise ValueError('must have at least one t_parameter to proceed.')
    # filter all logs
    if style is not None:
        logs = [log for log in logs if log['style'] == style]
    if type is not None:
        logs = [log for log in logs if log['type'] == type]
    if dataset is not None:
        logs = [log for log in logs if log['dataset'] == dataset]
    if qubits is not None:
        logs = [log for log in logs if log['n_qubits'] == qubits]
    if reuploads is not None:
        logs = [log for log in logs if log['n_reuploads'] == reuploads]
    if layers is not None:
        logs = [log for log in logs if log['n_layers'] == layers]
    # make all plot data
    plot_data = {}
    for log in logs:
        netname = ''
        if style is not None:
            pass
        else:
            netname += log['style'] + '_'
        if type is not None:
            pass
        else:
            netname += log['type'] + '_'
        if dataset is not None:
            pass
        else:
            netname += log['dataset'] + '_'
        if qubits is not None:
            pass
        else:
            netname += str(log['n_qubits']) + 'q_'
        if reuploads is not None:
            pass
        else:
            netname += str(log['n_reuploads']) + 'r_'
        if layers is not None:
            pass
        else:
            netname += str(log['n_layers']) + 'l'
        if netname not in plot_data:
            plot_data[netname] = {
                'n_classes': [],
                'accruacies': [],
                'treble': False
            }
            # see if this net should be trebled
            # if this net meets all the same t_parameters that are not None
            # then it should be trebled
            if t_style is not None:
                if log['style'] != t_style:
                    continue
            if t_type is not None:
                if log['type'] != t_type:
                    continue
            if t_dataset is not None:
                if log['dataset'] != t_dataset:
                    continue
            if t_qubits is not None:
                if log['n_qubits'] != t_qubits:
                    continue
            if t_reuploads is not None:
                if log['n_reuploads'] != t_reuploads:
                    continue
            if t_layers is not None:
                if log['n_layers'] != t_layers:
                    continue
            plot_data[netname]['treble'] = True
        plot_data[netname]['n_classes'].append(log['n_classes'])
        plot_data[netname]['accruacies'].append(max(log['test_accruacies']))
    # sort n_classes, and sort accruacies according to n_classes
    for netname in plot_data:
        plot_data[netname]['n_classes'], plot_data[netname]['accruacies'] = \
            zip(*sorted(zip(plot_data[netname]['n_classes'], plot_data[netname]['accruacies'])))
    # draw plot, and see if it should be trebled
    plt.figure(figsize=figsize)
    # make title
    title = 'Accuracy to number of classes: '
    filename = 'ACC_NCLS_'
    if style is not None:
        title += style + ' '
        filename += style + '_'
    if type is not None:
        title += type + ' '
        filename += type + '_'
    if dataset is not None:
        title += dataset + ' '
        filename += dataset + '_'
    if qubits is not None:
        title += str(qubits) + 'qubits '
        filename += str(qubits) + 'q_'
    if reuploads is not None:
        title += str(reuploads) + 'reuploads '
        filename += str(reuploads) + 'r_'
    if layers is not None:
        title += str(layers) + 'layers'
        filename += str(layers) + 'l'
    filename += '_T_' 
    if t_style is not None:
        filename += t_style + '_'
    if t_type is not None:
        filename += t_type + '_'
    if t_dataset is not None:
        filename += t_dataset + '_'
    if t_qubits is not None:
        filename += str(t_qubits) + 'q_'
    if t_reuploads is not None:
        filename += str(t_reuploads) + 'r_'
    if t_layers is not None:
        filename += str(t_layers) + 'l'
    plt.title(title)
    plt.xlabel('Number of classes')
    plt.ylabel('Accuracy')
    # plot
    for i, netname in enumerate(plot_data):
        if plot_data[netname]['treble']:
            plt.plot(plot_data[netname]['n_classes'], plot_data[netname]['accruacies'],
                     marker=markers[i%len(markers)], label=netname, alpha=1)
        else:
            plt.plot(plot_data[netname]['n_classes'], plot_data[netname]['accruacies'],
                     marker=markers[i%len(markers)], label=netname, alpha=0.3)
    plt.legend()
    plt.savefig(save_path + filename + '.pdf')
    plt.show()

