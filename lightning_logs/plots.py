import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import yaml
def read_params(file):
    # reads hparams.yaml file and returns params as dict

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    return params
def parse_params(params):
    if params['criterion'] == 'bce':
        if params['n_output'] > 1:
           return 'multi'
        else:
            return 'binary'
        
    return 'regression'


def plot(file : str , keys : list = ['val_loss_epoch','val_acc_epoch',] ):
    try:
        df = pd.read_csv( file + '/metrics.csv')
        params = read_params(file + '/hparams.yaml')
    except FileNotFoundError:
        print('FileNotFoundError')
        return

    x = 'regression'
    print(params)
    print(df.columns)
    if (x := parse_params(params) ) == 'multi' :

        list = [df[k].dropna() for k in keys]
    if (x := parse_params(params) ) == 'binary' :
        list = [df['val_acc_epoch'].dropna(),  df['val_loss_epoch'].dropna()]
    
    if (x := parse_params(params) ) == 'regression' :
        list = [df['val_loss_epoch'].dropna(), df['val_mae_epoch'].dropna() ]

    
    return x, params, list

def plot_dict(dict,num_rows=2):
    for model_type in dict:
        if model_type == 'regression':

            ylabels=['Loss (MSE)','MAE']
        else:
            ylabels=['Loss (BCE)','Accuracy']

        try:
            num_plots = len(list(dict[model_type].values())[0][0])
        except:
            continue
        fig,ax = plt.subplots(num_plots,1,figsize=(10,10))
        fig2,ax2 = plt.subplots(num_plots,1,figsize=(10,10))

        
        handles = []
        for i,threshold in enumerate(dict[model_type]):
            for t in range(len(dict[model_type][threshold])):
                for j,metric in enumerate(dict[model_type][threshold][t]):
                    if t:
                        x = ax2
                    else:
                        x = ax

                    try:
                        h = x[j].plot(metric,label= threshold.replace('_',' ' + 'km labels') )
                    except:
                        continue

                    x[j].set_ylabel(ylabels[j])
                    x[j].set_xlabel('Epoch')
                    x[j].grid()
        labels = [ item + ' km threshold label' for item in list(dict[model_type].keys())]
        fig.legend(labels,loc='upper center', ncol=4)
        fig2.legend(labels,loc='upper center', ncol=4)

        fig.savefig('plots/models/' +model_type + '.pdf',bbox_inches='tight',dpi=300)
        fig2.savefig('plots/models/temporal_' +model_type + '.pdf',bbox_inches='tight',dpi=300)
        

        plt.close()
import os

def parse_dir_to_dict(metricsdir: str):
    dirs = os.listdir(metricsdir)
    dirs.sort()
    threshold_dict = {'multi' : {},'regression' : {},'binary' : {}}
    
    for dir in dirs:
        if dir.startswith('version'):
            try:
                type, params, l  =plot(f'{metricsdir}/' + dir )
            except :
                continue
                
            thresh = params['threshold'].split('_')[1]
            if thresh not in threshold_dict[type]:
                threshold_dict[type][thresh] = [[0],[0]]

            threshold_dict[type][thresh][params['temporal']] = l
        

    # print(threshold_dict['multi'].keys())
    # plot_dict(threshold_dict)
    return threshold_dict

#pylint: disable=consider-using-dict-items

if __name__ == "__main__":
    dir1 = parse_dir_to_dict('lightning_logs/run_1_final')
    dir2 = parse_dir_to_dict('lightning_logs/run_2_final') 
    dir3 = parse_dir_to_dict('lightning_logs/run_3_final')

    print(dir1)
    print(dir2)
    print(dir3)
    
    l = [dir1,dir2,dir3]
    average_pd_frame = pd.DataFrame()
    for model_type in dir1:
        for threshold in dir1[model_type]:
            for t in range(len(dir1[model_type][threshold])):
                for j,metric in enumerate(dir1[model_type][threshold][t]):
                    metric1 = dir1[model_type][threshold][t][j].values

                    try:
                        metric2 = dir2[model_type][threshold][t][j].values
                    except:
                        metric2 = dir1[model_type][threshold][t][j].values # missing value?
                    metric3 = dir3[model_type][threshold][t][j].values

                    arr = np.c_[metric1,metric2,metric3]

                    dir1[model_type][threshold][t][j] = np.mean(arr,axis=1)
   
    plot_dict(dir1)

    




