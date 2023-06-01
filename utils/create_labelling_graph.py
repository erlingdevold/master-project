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

    plt.ylabel('Size of labels')
    plt.xlabel('Number of examples')

    plt.legend()
    plt.savefig("plots/size_labels.pdf")

def distinct_classes_graph(dir:str):
    threshold_dict = {1:[],5:[],10:[],20:[] }

    for d ,threshold in load_dir(dir):
        if len(d.keys()) > 0:
            
            threshold_dict[threshold].append(len(d.keys()))
    
    for k in threshold_dict.keys():
        threshold_dict[k] = sorted(threshold_dict[k], key=lambda x: x)
        threshold_dict[k] = np.array(threshold_dict[k])
        plt.plot(threshold_dict[k],label=f'threshold={k}')

    plt.ylabel('Number of distinct classes')
    plt.xlabel('Number of examples')

    plt.legend()
    plt.savefig("plots/distinct_classes.pdf")
import datetime

def to_utc(dt):
    """Convert a datetime object to UTC time."""
    return dt.astimezone(datetime.timezone.utc)

def from_string(string : str,fmt="%d.%m.%Y"):
    """Convert a string to a datetime object."""
    return datetime.datetime.strptime(string,fmt) 


def create_delta_time(truth :str, obj):
    """Create a delta time object."""
    if obj is []:
        return []
    truth = from_string(truth,fmt="D%Y%m%d")
    return [(to_utc(from_string(x)) - to_utc(truth)).days for x in obj] 

def retrieve_date_dict(image_dir:str , dir:str='ds/labels_crimac_2021'):
    threshold_dict = {1:[],5:[],10:[],20:[] }
    date_dict = {1:[],5:[],10:[],20:[]}

    for d ,threshold in load_dir(dir):
        for species in d:
            threshold_dict[threshold].append((d[species]['date']))
    
    for date in os.listdir(image_dir):
        if date.endswith('.nc'):
            truth = date.split('-')[1]

            for k in threshold_dict.keys():
                for t in threshold_dict[k]:
                    for date in t:
                        date_dict[k].append((to_utc(from_string(date)) - to_utc(from_string(truth,fmt="D%Y%m%d"))).days)
                    date_dict[k].append(create_delta_time(date,t)) 

def temporal_proximity_histogram(image_dir: str,dir:str='ds/labels_crimac_2021'):
    # threshold_dict = {1:[],5:[],10:[],20:[] }

    # for d ,threshold in load_dir(dir):
    #     for species in d:
    #         threshold_dict[threshold].append((d[species]['date']))
    
    # date_dict = {1:[],5:[],10:[],20:[]}
    # for date in os.listdir(image_dir):
    #     if date.endswith('.nc'):
    #         truth = date.split('-')[1]

    #         for k in threshold_dict.keys():
    #             for t in threshold_dict[k]:
    #                 for date in t:
    #                     try:
    #                         date_dict[k].append((to_utc(from_string(date)) - to_utc(from_string(truth,fmt="D%Y%m%d"))).days)
    #                     except ValueError:
    #                         print('oki')
    #                         continue
                    # date_dict[k].append(create_delta_time(date,t))

    date_dict = json.load(open('plots/date_dict.json','r'))


    fig,ax = plt.subplots(4,1,figsize=(10,10))

    ax = ax.flatten()
    i = 0
    for k in date_dict.keys():
        date_dict[k] = sorted(date_dict[k], key=lambda x: x)
        date_dict[k] = np.array(date_dict[k])
        ax[i].hist(date_dict[k],bins=1000,label=f'threshold={k}')
        ax[i].set_ylabel('Number of catch messages')
        i += 1

    fig.legend(['1 km threshold','5 km threshold','10 km threshold','20 km threshold'],loc='upper center', ncol=2)
    ax[-1].set_xlabel('Temporal proximity to truth (days since truth)')

    plt.savefig(f"plots/temporal_proximity.pdf")

            
def compute_time(dir: str = 'ds/stats'):
    threshold_dict = {1:[],5:[],10:[],20:[] }
    for d ,_ in load_dir(dir):
        print(d)
        if d['size'] < 0 or d['time'] < 0:
            continue
        threshold_dict[d['threshold']].append((d['time'],d['size']))
    avg_time = []
    
    for k in threshold_dict.keys():
        threshold_dict[k] = sorted(threshold_dict[k], key=lambda x: x[0])
        threshold_dict[k] = np.array(threshold_dict[k])
        plt.scatter(threshold_dict[k][:,0],threshold_dict[k][:,1], label=f'threshold={k}')
        avg_time.append(np.mean(threshold_dict[k][:,0]))
    
    print(np.mean(avg_time))

    
    # plt.xscale('log')
    plt.xlabel('Time to label (s)')
    plt.ylabel('Number of examples')

    plt.legend()
    plt.savefig("plots/time_labels.pdf")





if __name__ == "__main__":
    # size_graph('ds/stats')
    # plt.clf()
    # distinct_classes_graph('ds/labels_crimac_2021/')
    # plt.clf()
    # temporal_proximity_histogram('ds/ds_unlabeled/','ds/labels_crimac_2021/')
    compute_time()
