import matplotlib.pyplot as plt

import numpy as np


import json,os
def load_json(fn : str):
    with open(fn, 'r') as f:
        return json.load(f)


def load_dir(dir : str):
    for fn in os.listdir(dir):
        if fn.endswith('.json'):
            try:
                threshold = int(fn.split('_')[1].split('.')[0])
            except:
                threshold = 0
            yield load_json(os.path.join(dir, fn)), threshold


def size_graph(dir : str):
    threshold_dict = {1:[],5:[],10:[],20:[] }

    for d ,_ in load_dir(dir):
        print(d)
        if d['size'] < 0 or d['time'] < 0:
            continue
        threshold_dict[d['threshold']].append((d['time'],d['size']))
    
    for k in threshold_dict.keys():
        threshold_dict[k] = sorted(threshold_dict[k], key=lambda x: x[1])
        threshold_dict[k] = np.array(threshold_dict[k])
        plt.plot(threshold_dict[k][:,1],label=f'threshold={k}')

    plt.xlabel('Size of labels')
    plt.ylabel('Number of examples')

    plt.legend()
    plt.savefig("plots/size_labels.pdf")

def distinct_classes_graph(dir:str):
    threshold_dict = {1:[],5:[],10:[],20:[] }

    for d ,threshold in load_dir(dir):
        threshold_dict[threshold].append(len(d.keys()))
    
    for k in threshold_dict.keys():
        threshold_dict[k] = sorted(threshold_dict[k], key=lambda x: x)
        threshold_dict[k] = np.array(threshold_dict[k])
        plt.plot(threshold_dict[k],label=f'threshold={k}')

    plt.xlabel('Number of distinct classes')
    plt.ylabel('Number of examples')

    plt.legend()
    plt.savefig("plots/distinct_classes.pdf")


if __name__ == "__main__":
    size_graph('ds/stats')
    plt.clf()
    distinct_classes_graph('ds/labels_crimac_2021/')
